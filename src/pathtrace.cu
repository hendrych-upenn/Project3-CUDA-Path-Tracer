#include "pathtrace.h"

#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>
#include <thrust/partition.h>
#include <thrust/device_vector.h>
#include <limits.h>

#include "sceneStructs.h"
#include "scene.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "utilities.h"
#include "intersections.h"
#include "interactions.h"
#include <iostream>

#define CONTIGUOUS_MATERIALS 1

#define ERRORCHECK 1

#define FILENAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#define checkCUDAError(msg) checkCUDAErrorFn(msg, FILENAME, __LINE__)
void checkCUDAErrorFn(const char* msg, const char* file, int line)
{
#if ERRORCHECK
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess == err)
    {
        return;
    }

    fprintf(stderr, "CUDA error");
    if (file)
    {
        fprintf(stderr, " (%s:%d)", file, line);
    }
    fprintf(stderr, ": %s: %s\n", msg, cudaGetErrorString(err));
#ifdef _WIN32
    getchar();
#endif // _WIN32
    exit(EXIT_FAILURE);
#endif // ERRORCHECK
}

__host__ __device__
thrust::default_random_engine makeSeededRandomEngine(int iter, int index, int depth)
{
    int h = utilhash((1 << 31) | (depth << 22) | iter) ^ utilhash(index);
    return thrust::default_random_engine(h);
}

//Kernel that writes the image to the OpenGL PBO directly.
__global__ void sendImageToPBO(uchar4* pbo, glm::ivec2 resolution, int iter, glm::vec3* image)
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < resolution.x && y < resolution.y)
    {
        int index = x + (y * resolution.x);
        glm::vec3 pix = image[index];

        glm::ivec3 color;
        color.x = glm::clamp((int)(pix.x / iter * 255.0), 0, 255);
        color.y = glm::clamp((int)(pix.y / iter * 255.0), 0, 255);
        color.z = glm::clamp((int)(pix.z / iter * 255.0), 0, 255);

        // Each thread writes one pixel location in the texture (textel)
        pbo[index].w = 0;
        pbo[index].x = color.x;
        pbo[index].y = color.y;
        pbo[index].z = color.z;
    }
}

static Scene* hst_scene = NULL;
static GuiDataContainer* guiData = NULL;
static glm::vec3* dev_image = NULL;
static Geom* dev_geoms = NULL;
static Material* dev_materials = NULL;
static PathSegment* dev_paths = NULL;
static ShadeableIntersection* dev_intersections = NULL;
// TODO: static variables for device memory, any extra info you need, etc
// ...
static uint8_t* dev_bufferBytes;
static size_t* dev_bufferOffsets;
static uint8_t* dev_imageBufferBytes;
static size_t* dev_imageBufferOffsets;
static ImageData* dev_imagesData;

void InitDataContainer(GuiDataContainer* imGuiData)
{
    guiData = imGuiData;
}

void pathtraceInit(Scene* scene)
{
    hst_scene = scene;

    const Camera& cam = hst_scene->state.cameras[hst_scene->state.activeCamera];
    const int pixelcount = cam.resolution.x * cam.resolution.y;

    cudaMalloc(&dev_image, pixelcount * sizeof(glm::vec3));
    cudaMemset(dev_image, 0, pixelcount * sizeof(glm::vec3));

    cudaMalloc(&dev_paths, pixelcount * sizeof(PathSegment));

    cudaMalloc(&dev_geoms, scene->geoms.size() * sizeof(Geom));
    cudaMemcpy(dev_geoms, scene->geoms.data(), scene->geoms.size() * sizeof(Geom), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_materials, scene->materials.size() * sizeof(Material));
    cudaMemcpy(dev_materials, scene->materials.data(), scene->materials.size() * sizeof(Material), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_intersections, pixelcount * sizeof(ShadeableIntersection));
    cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

    // TODO: initialize any extra device memeory you need
    cudaMalloc(&dev_bufferBytes, scene->bufferBytes.size() * sizeof(uint8_t));
    cudaMemcpy(dev_bufferBytes, scene->bufferBytes.data(), scene->bufferBytes.size() * sizeof(uint8_t), cudaMemcpyHostToDevice);
    
    cudaMalloc(&dev_bufferOffsets, scene->bufferOffsets.size() * sizeof(size_t));
    cudaMemcpy(dev_bufferOffsets, scene->bufferOffsets.data(), scene->bufferOffsets.size() * sizeof(size_t), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_imageBufferBytes, scene->imageBufferBytes.size() * sizeof(uint8_t));
    cudaMemcpy(dev_imageBufferBytes, scene->imageBufferBytes.data(), scene->imageBufferBytes.size() * sizeof(uint8_t), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_imageBufferOffsets, scene->imageBufferOffsets.size() * sizeof(size_t));
    cudaMemcpy(dev_imageBufferOffsets, scene->imageBufferOffsets.data(), scene->imageBufferOffsets.size() * sizeof(size_t), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_imagesData, scene->imagesData.size() * sizeof(ImageData));
    cudaMemcpy(dev_imagesData, scene->imagesData.data(), scene->imagesData.size() * sizeof(ImageData), cudaMemcpyHostToDevice);

    checkCUDAError("pathtraceInit");
}

void pathtraceFree()
{
    cudaFree(dev_image);  // no-op if dev_image is null
    cudaFree(dev_paths);
    cudaFree(dev_geoms);
    cudaFree(dev_materials);
    cudaFree(dev_intersections);
    // TODO: clean up any extra device memory you created
    cudaFree(dev_bufferBytes);
    cudaFree(dev_bufferOffsets);
    cudaFree(dev_imageBufferBytes);
    cudaFree(dev_imageBufferOffsets);
    cudaFree(dev_imagesData);
    checkCUDAError("pathtraceFree");
}

/**
* Generate PathSegments with rays from the camera through the screen into the
* scene, which is the first bounce of rays.
*
* Antialiasing - add rays for sub-pixel sampling
* motion blur - jitter rays "in time"
* lens effect - jitter ray origin positions based on a lens
*/
__global__ void generateRayFromCamera(Camera cam, int iter, int traceDepth, PathSegment* pathSegments, bool useAntiAliasing)
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < cam.resolution.x && y < cam.resolution.y) {
        int index = x + (y * cam.resolution.x);
        PathSegment& segment = pathSegments[index];

        segment.ray.origin = cam.position;
        segment.color = glm::vec3(1.0f, 1.0f, 1.0f);
        segment.accumulatedColor = glm::vec3(0.0);

        float xrespos = ((float)x - (float)cam.resolution.x * 0.5f );
        float yrespos = ((float)y - (float)cam.resolution.y * 0.5f);
        if (useAntiAliasing) {
            thrust::default_random_engine rng = makeSeededRandomEngine(iter, index, traceDepth);
            thrust::uniform_real_distribution<float> u01(0, 1);
            xrespos += u01(rng);
            yrespos += u01(rng);
        }
        segment.ray.direction = glm::normalize(cam.view
            - cam.right * cam.pixelLength.x * xrespos
            - cam.up * cam.pixelLength.y * yrespos
        );

        segment.pixelIndex = index;
        segment.remainingBounces = traceDepth;
    }
}

// TODO:
// computeIntersections handles generating ray intersections ONLY.
// Generating new rays is handled in your shader(s).
// Feel free to modify the code below.
__global__ void computeIntersections(
    int depth,
    int num_paths,
    const PathSegment* pathSegments,
    const Geom* geoms,
    const uint8_t* bufferBytes,
    const size_t* bufferOffsets,
    int geoms_size,
    ShadeableIntersection* intersections)
{
    int path_index = blockIdx.x * blockDim.x + threadIdx.x;

    if (path_index < num_paths)
    {
        PathSegment pathSegment = pathSegments[path_index];

        float t = 0.0f;
        glm::vec3 intersect_point;
        glm::vec3 normal;
        glm::vec4 tangent;
        glm::vec2 texCoords[MAX_PATHTRACE_TEXTURES];

        float t_min = FLT_MAX;
        int hit_geom_index = -1;
        bool outside = true;

        glm::vec3 tmp_intersect;
        glm::vec3 tmp_normal;
        glm::vec4 tmp_tangent;
        glm::vec2 tmp_texCoords[MAX_PATHTRACE_TEXTURES];

        // naive parse through global geoms

        for (int i = 0; i < geoms_size; i++)
        {
            const Geom& geom = geoms[i];

            if (geom.type == CUBE)
            {
                t = boxIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
            }
            else if (geom.type == SPHERE)
            {
                t = sphereIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
            }
            else if (geom.type == MESH)
            {
                t = meshIntersectionTest(geom, pathSegment.ray, bufferBytes, bufferOffsets, tmp_intersect, tmp_normal, 
                    tmp_tangent, tmp_texCoords
                );
            }
            // TODO: add more intersection tests here... triangle? metaball? CSG?

            // Compute the minimum t from the intersection tests to determine what
            // scene geometry object was hit first.
            if (t > 0.0f && t_min > t)
            {
                t_min = t;
                hit_geom_index = i;
                intersect_point = tmp_intersect;
                normal = tmp_normal;
                tangent = tmp_tangent;
#pragma unroll
                for (int j = 0; j < MAX_PATHTRACE_TEXTURES; j++) {
                    texCoords[j] = tmp_texCoords[j];
                }
            }
        }

        if (hit_geom_index == -1)
        {
            intersections[path_index].t = -1.0f;
#if CONTIGUOUS_MATERIALS
            intersections[path_index].materialId = INT_MAX;
#endif
        }
        else
        {
            // The ray hits something
            intersections[path_index].t = t_min;
            intersections[path_index].materialId = geoms[hit_geom_index].materialid;
            intersections[path_index].surfaceNormal = normal;
            intersections[path_index].surfaceTangent = tangent;
#pragma unroll
            for (int j = 0; j < MAX_PATHTRACE_TEXTURES; j++) {
                getTexCoords(intersections[path_index], j) = texCoords[j];
            }
        }
    }
}

__host__ __device__ glm::vec4 indexIntoImage(
    const uint8_t* imageBufferBytes,
    const size_t* imageBufferOffsets,
    const ImageData* imagesData,
    int imageBufferIdx,
    glm::vec2 texCoords
) {
    ImageData imageData = imagesData[imageBufferIdx];
    // TODO: update for different sampling modes and wrapping
    texCoords = glm::fract(texCoords);
    int s = glm::floor(texCoords.s * imageData.width);
    int t = glm::floor(texCoords.t * imageData.height);

    size_t imageBufferOffset = imageBufferOffsets[imageBufferIdx];
    imageBufferOffset += (s + ((size_t)t) * imageData.width) * ((((size_t)imageData.bitDepthPerChannel) * imageData.component) / 8);

    const uint8_t* pixelData = imageBufferBytes + imageBufferOffset;

    return glm::vec4(pixelData[0], pixelData[1], pixelData[2], pixelData[3]);

}

// LOOK: "fake" shader demonstrating what you might do with the info in
// a ShadeableIntersection, as well as how to use thrust's random number
// generator. Observe that since the thrust random number generator basically
// adds "noise" to the iteration, the image should start off noisy and get
// cleaner as more iterations are computed.
//
// Note that this shader does NOT do a BSDF evaluation!
// Your shaders should handle that - this can allow techniques such as
// bump mapping.
__global__ void shadeMaterial(
    int iter,
    int num_paths,
    int depth,
    const ShadeableIntersection* shadeableIntersections,
    PathSegment* pathSegments,
    const uint8_t* imageBufferBytes,
    const size_t* imageBufferOffsets,
    const ImageData* imagesData,
    const Material* materials)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_paths)
    {
        ShadeableIntersection intersection = shadeableIntersections[idx];
        if (intersection.t > 0.0f) // if the intersection exists...
        {
            PathSegment pathSegment = pathSegments[idx];
            // Set up the RNG
            // LOOK: this is how you use thrust's RNG! Please look at
            // makeSeededRandomEngine as well.
            thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, depth);
            thrust::uniform_real_distribution<float> u01(0, 1);

            Material material = materials[intersection.materialId];
            glm::vec3 materialColor = material.color;
            glm::vec3 normal = intersection.surfaceNormal;

            if (material.baseColorTexture.exists) {
                auto& baseColorTexture = material.baseColorTexture;
                glm::vec2 texCoords = getTexCoords(intersection, baseColorTexture.texCoordsIdx);

                glm::vec3 texColor = glm::vec3(indexIntoImage(imageBufferBytes, imageBufferOffsets, imagesData, baseColorTexture.imageBufferIdx, texCoords)) / 255.0f;

                materialColor *= texColor;
            }

            if (material.normalTexture.exists) {
                auto& normalTexture = material.normalTexture;
                float scale = material.normalScale;

                glm::vec2 texCoords = getTexCoords(intersection, normalTexture.imageBufferIdx);

                glm::vec3 texColor = glm::vec3(indexIntoImage(imageBufferBytes, imageBufferOffsets, imagesData, normalTexture.imageBufferIdx, texCoords)) / 255.0f;

                texColor = texColor * 2.0f - 1.0f; // map to [-1, 1]

                glm::vec3 surfaceNormal = glm::normalize(intersection.surfaceNormal);
                glm::vec3 surfaceTangent = glm::vec3(intersection.surfaceTangent);
                surfaceTangent = glm::normalize(surfaceTangent - glm::dot(surfaceNormal, surfaceTangent) * surfaceNormal);
                glm::vec3 surfaceBitangent = glm::normalize(glm::cross(surfaceNormal, surfaceTangent) * intersection.surfaceTangent.w);
                //pathSegment.accumulatedColor = (surfaceBitangent + glm::vec3(1.0f)) / 2.0f;

                normal = glm::normalize(texColor.x * scale * surfaceTangent + texColor.y * scale * surfaceBitangent + texColor.z * surfaceNormal);
            }

            pathSegment.color *= materialColor;

            // If the material indicates that the object was a light, "light" the ray
            pathSegment.accumulatedColor += pathSegment.color * material.emittance;
            //pathSegment.accumulatedColor = (intersection.surfaceNormal + glm::vec3(1.0f)) / 2.0f;
            //pathSegment.accumulatedColor = (normal + glm::vec3(1.0f)) / 2.0f;
            //pathSegment.accumulatedColor = (glm::vec3(intersection.surfaceTangent) + glm::vec3(1.0f)) / 2.0f;
            //pathSegment.accumulatedColor = glm::vec3(1.0);
            //pathSegment.accumulatedColor = pathSegment.color * (intersection.t / 1.0f);
            
            // Otherwise, do some pseudo-lighting computation. This is actually more
            // like what you would expect from shading in a rasterizer like OpenGL.
            // TODO: replace this! you should be able to start with basically a one-liner
            
            glm::vec3 intersectionPoint = getPointOnRay(pathSegment.ray, intersection.t);
            //pathSegment.accumulatedColor = (intersectionPoint / 1.0f); //  pathSegment.color *
            //pathSegment.accumulatedColor = glm::vec3(depth / 2.0f); //  pathSegment.color *
            if (depth == 1) {
                //pathSegment.accumulatedColor += glm::vec3(1.0f, 0.0f, 0.0f);
            }
            if (depth == 2) {
                //pathSegment.accumulatedColor += glm::vec3(0.0f, 1.0f, 0.0f);
                //pathSegment.accumulatedColor += glm::vec3(1.0f);
            }
            scatterRay(pathSegment, intersectionPoint, normal, material, rng);
            //pathSegment.accumulatedColor = (pathSegment.ray.direction + glm::vec3(1.0f))/2.0f; //  pathSegment.color *
            pathSegment.remainingBounces -= 1;
            //float lightTerm = glm::dot(intersection.surfaceNormal, glm::vec3(0.0f, 1.0f, 0.0f));
            //pathSegments[idx].color *= (materialColor * lightTerm) * 0.3f + ((1.0f - intersection.t * 0.02f) * materialColor) * 0.7f;
            //pathSegments[idx].color *= u01(rng); // apply some noise because why not
            
            pathSegments[idx] = pathSegment;
        }
        else {
            // If there was no intersection, color the ray black.
            // Lots of renderers use 4 channel color, RGBA, where A = alpha, often
            // used for opacity, in which case they can indicate "no opacity".
            // This can be useful for post-processing and image compositing.
            pathSegments[idx].color = glm::vec3(0.0f);
        }
    }
}

// Add the current iteration's output to the overall image
__global__ void finalGather(int nPaths, glm::vec3* image, PathSegment* iterationPaths, int incomplete_paths)
{
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (index < nPaths)
    {
        PathSegment iterationPath = iterationPaths[index];
        image[iterationPath.pixelIndex] += iterationPath.accumulatedColor;
    }
}

struct does_intersect_functor
{
    __host__ __device__
        bool operator()(const ShadeableIntersection& intersection)
    {
        return intersection.t > 0.0;
    }
};

struct material_ordering_functor 
{
    __host__ __device__
        bool operator()(const ShadeableIntersection& intersectionA, const ShadeableIntersection& intersectionB)
    {
        return intersectionA.materialId < intersectionB.materialId;
    }

};

struct does_not_intersect_functor
{
    __host__ __device__
        bool operator()(const ShadeableIntersection& intersection)
    {
        return intersection.t < 0.0;
    }
};


/**
 * Wrapper for the __global__ call that sets up the kernel calls and does a ton
 * of memory management
 */
void pathtrace(uchar4* pbo, int frame, int iter, bool useAntiAliasing)
{
    const int traceDepth = hst_scene->state.traceDepth;
    const Camera& cam = hst_scene->state.cameras[hst_scene->state.activeCamera];
    const int pixelcount = cam.resolution.x * cam.resolution.y;

    // 2D block for generating ray from camera
    const dim3 blockSize2d(8, 8);
    const dim3 blocksPerGrid2d(
        (cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
        (cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);

    // 1D block for path tracing
    const int blockSize1d = 128;

    ///////////////////////////////////////////////////////////////////////////

    // Recap:
    // * Initialize array of path rays (using rays that come out of the camera)
    //   * You can pass the Camera object to that kernel.
    //   * Each path ray must carry at minimum a (ray, color) pair,
    //   * where color starts as the multiplicative identity, white = (1, 1, 1).
    //   * This has already been done for you.
    // * For each depth:
    //   * Compute an intersection in the scene for each path ray.
    //     A very naive version of this has been implemented for you, but feel
    //     free to add more primitives and/or a better algorithm.
    //     Currently, intersection distance is recorded as a parametric distance,
    //     t, or a "distance along the ray." t = -1.0 indicates no intersection.
    //     * Color is attenuated (multiplied) by reflections off of any object
    //   * TODO: Stream compact away all of the terminated paths.
    //     You may use either your implementation or `thrust::remove_if` or its
    //     cousins.
    //     * Note that you can't really use a 2D kernel launch any more - switch
    //       to 1D.
    //   * TODO: Shade the rays that intersected something or didn't bottom out.
    //     That is, color the ray by performing a color computation according
    //     to the shader, then generate a new ray to continue the ray path.
    //     We recommend just updating the ray's PathSegment in place.
    //     Note that this step may come before or after stream compaction,
    //     since some shaders you write may also cause a path to terminate.
    // * Finally, add this iteration's results to the image. This has been done
    //   for you.

    // TODO: perform one iteration of path tracing

    generateRayFromCamera<<<blocksPerGrid2d, blockSize2d>>>(cam, iter, traceDepth, dev_paths, useAntiAliasing);
    checkCUDAError("generate camera ray");

    int depth = 0;
    PathSegment* dev_path_end = dev_paths + pixelcount;
    int num_paths = dev_path_end - dev_paths;
    PathSegment* dev_unterminated_paths_end = dev_path_end;
    thrust::device_ptr<PathSegment> dev_thrust_paths(dev_paths);
    thrust::device_ptr<ShadeableIntersection> dev_thrust_intersections(dev_intersections);

    // --- PathSegment Tracing Stage ---
    // Shoot ray into scene, bounce between objects, push shading chunks

    bool iterationComplete = false;
    while (!iterationComplete)
    {
        //std::cout << "depth: " << depth << std::endl;
        // clean shading chunks
        cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));
        checkCUDAError("cudaMemset bounce");

        // tracing
        dim3 numblocksPathSegmentTracing = (num_paths + blockSize1d - 1) / blockSize1d;
        computeIntersections<<<numblocksPathSegmentTracing, blockSize1d>>> (
            depth,
            num_paths,
            dev_paths,
            dev_geoms,
            dev_bufferBytes,
            dev_bufferOffsets,
            hst_scene->geoms.size(),
            dev_intersections
        );
        checkCUDAError("trace one bounce");
        //cudaDeviceSynchronize();
        depth++;

        // TODO:
        // --- Shading Stage ---
        // Shade path segments based on intersections and generate new rays by
        // evaluating the BSDF.
        // Start off with just a big kernel that handles all the different
        // materials you have in the scenefile.
        // TODO: compare between directly shading the path segments and shading
        // path segments that have been reshuffled to be contiguous in memory.
#if CONTIGUOUS_MATERIALS
        thrust::sort_by_key(dev_thrust_intersections, dev_thrust_intersections + num_paths, dev_thrust_paths, material_ordering_functor());
        checkCUDAError("thrust::sort_by_key");
#endif
        shadeMaterial <<<numblocksPathSegmentTracing, blockSize1d>>>(
            iter,
            num_paths,
            depth,
            dev_intersections,
            dev_paths,
            dev_imageBufferBytes,
            dev_imageBufferOffsets,
            dev_imagesData,
            dev_materials
        );
        checkCUDAError("shadeMaterial");
#if CONTIGUOUS_MATERIALS
        thrust::device_ptr<ShadeableIntersection> dev_thrust_unterminated_intersections_end = thrust::find_if(dev_thrust_intersections, dev_thrust_intersections + num_paths, does_not_intersect_functor());
        checkCUDAError("thrust::find_if");
        dev_unterminated_paths_end = dev_paths + (dev_thrust_unterminated_intersections_end - dev_thrust_intersections);
#else
        thrust::device_ptr<PathSegment> dev_thrust_unterminated_paths_end = thrust::partition(dev_thrust_paths, dev_thrust_paths + num_paths, dev_thrust_intersections, does_intersect_functor());
        checkCUDAError("thrust::partition");
        dev_unterminated_paths_end = dev_thrust_unterminated_paths_end.get();
#endif
        num_paths = dev_unterminated_paths_end - dev_paths;

        iterationComplete = num_paths == 0 || depth >= traceDepth;

        if (guiData != NULL)
        {
            guiData->TracedDepth = depth;
        }
    }

    // Assemble this iteration and apply it to the image
    dim3 numBlocksPixels = (pixelcount + blockSize1d - 1) / blockSize1d;
    finalGather<<<numBlocksPixels, blockSize1d>>>(pixelcount, dev_image, dev_paths, num_paths);
    checkCUDAError("finalGather");

    ///////////////////////////////////////////////////////////////////////////

    // Send results to OpenGL buffer for rendering
    sendImageToPBO<<<blocksPerGrid2d, blockSize2d>>>(pbo, cam.resolution, iter, dev_image);
    checkCUDAError("sendImageToPBO");

    // Retrieve image from GPU
    cudaMemcpy(hst_scene->state.image.data(), dev_image,
        pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

    checkCUDAError("pathtrace");
}
