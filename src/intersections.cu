#include "intersections.h"

#define ARBITRARY_MESH_BOUNDING_VOLUME_CULLING 0

__host__ __device__ float boxIntersectionTest(
    Geom box,
    Ray r,
    glm::vec3 &intersectionPoint,
    glm::vec3 &normal,
    bool &outside)
{
    Ray q;
    q.origin    =                multiplyMV(box.inverseTransform, glm::vec4(r.origin   , 1.0f));
    q.direction = glm::normalize(multiplyMV(box.inverseTransform, glm::vec4(r.direction, 0.0f)));

    float tmin = -1e38f;
    float tmax = 1e38f;
    glm::vec3 tmin_n;
    glm::vec3 tmax_n;
    for (int xyz = 0; xyz < 3; ++xyz)
    {
        float qdxyz = q.direction[xyz];
        /*if (glm::abs(qdxyz) > 0.00001f)*/
        {
            float t1 = (-0.5f - q.origin[xyz]) / qdxyz;
            float t2 = (+0.5f - q.origin[xyz]) / qdxyz;
            float ta = glm::min(t1, t2);
            float tb = glm::max(t1, t2);
            glm::vec3 n;
            n[xyz] = t2 < t1 ? +1 : -1;
            if (ta > 0 && ta > tmin)
            {
                tmin = ta;
                tmin_n = n;
            }
            if (tb < tmax)
            {
                tmax = tb;
                tmax_n = n;
            }
        }
    }

    if (tmax >= tmin && tmax > 0)
    {
        outside = true;
        if (tmin <= 0)
        {
            tmin = tmax;
            tmin_n = tmax_n;
            outside = false;
        }
        intersectionPoint = multiplyMV(box.transform, glm::vec4(getPointOnRay(q, tmin), 1.0f));
        normal = glm::normalize(multiplyMV(box.invTranspose, glm::vec4(tmin_n, 0.0f)));
        return glm::length(r.origin - intersectionPoint);
    }

    return -1;
}

__host__ __device__ bool simpleBoxIntersectionTest(
    glm::vec3 minCorner,
    glm::vec3 maxCorner,
    glm::vec3 ro,
    glm::vec3 rd)
{
    float tmin = -1e38f;
    float tmax = 1e38f;
#pragma unroll
    for (int xyz = 0; xyz < 3; ++xyz)
    {
        float rdxyz = rd[xyz];
        float roxyz = ro[xyz];
        /*if (glm::abs(qdxyz) > 0.00001f)*/
        {
            float t1 = (minCorner[xyz] - roxyz) / rdxyz;
            float t2 = (maxCorner[xyz] - roxyz) / rdxyz;
            float ta = glm::min(t1, t2);
            float tb = glm::max(t1, t2);
            if (ta > 0 && ta > tmin)
            {
                tmin = ta;
            }
            if (tb < tmax)
            {
                tmax = tb;
            }
        }
    }
    return tmax >= tmin && tmax > 0;
}

__host__ __device__ float sphereIntersectionTest(
    Geom sphere,
    Ray r,
    glm::vec3 &intersectionPoint,
    glm::vec3 &normal,
    bool &outside)
{
    float radius = .5;

    glm::vec3 ro = multiplyMV(sphere.inverseTransform, glm::vec4(r.origin, 1.0f));
    glm::vec3 rd = glm::normalize(multiplyMV(sphere.inverseTransform, glm::vec4(r.direction, 0.0f)));

    Ray rt;
    rt.origin = ro;
    rt.direction = rd;

    float vDotDirection = glm::dot(rt.origin, rt.direction);
    float radicand = vDotDirection * vDotDirection - (glm::dot(rt.origin, rt.origin) - powf(radius, 2));
    if (radicand < 0)
    {
        return -1;
    }

    float squareRoot = sqrt(radicand);
    float firstTerm = -vDotDirection;
    float t1 = firstTerm + squareRoot;
    float t2 = firstTerm - squareRoot;

    float t = 0;
    if (t1 < 0 && t2 < 0)
    {
        return -1;
    }
    else if (t1 > 0 && t2 > 0)
    {
        t = min(t1, t2);
        outside = true;
    }
    else
    {
        t = max(t1, t2);
        outside = false;
    }

    glm::vec3 objspaceIntersection = getPointOnRay(rt, t);

    intersectionPoint = multiplyMV(sphere.transform, glm::vec4(objspaceIntersection, 1.f));
    normal = glm::normalize(multiplyMV(sphere.invTranspose, glm::vec4(objspaceIntersection, 0.f)));
    if (!outside)
    {
        normal = -normal;
    }

    return glm::length(r.origin - intersectionPoint);
}

template <typename T>
__host__ __device__ T getData(int idx, FlatBufferView flatBufferView,
    const uint8_t* bufferBytes,
    const size_t* bufferOffsets) {
    size_t bufferOffset = bufferOffsets[flatBufferView.buffer];
    const uint8_t* bufferStart = bufferBytes + bufferOffset;
    const uint8_t* dataStart = bufferStart + flatBufferView.byteOffset;
    size_t stride = flatBufferView.byteStride == 0 ? sizeof(T) : flatBufferView.byteStride;
    const uint8_t* index_start = dataStart + stride * idx;
    return *reinterpret_cast<const T*>(index_start);
}

template <typename T>
__host__ __device__ T interpolateBarycentric(glm::vec3 baryCoords, T d0, T d1, T d2) {
    return (1.0f - baryCoords.x - baryCoords.y) * d0 + baryCoords.x * d1 + baryCoords.y * d2;
}

__host__ __device__ float meshIntersectionTest(
    const Geom mesh,
    const Ray r,
    const uint8_t* bufferBytes,
    const size_t* bufferOffsets,
    glm::vec3& intersectionPoint,
    glm::vec3& normal
    //,std::array<glm::vec2, MAX_PATHTRACE_TEXTURES>& texCoords
) {
    glm::vec3 ro = multiplyMV(mesh.inverseTransform, glm::vec4(r.origin, 1.0f));
    glm::vec3 rd = glm::normalize(multiplyMV(mesh.inverseTransform, glm::vec4(r.direction, 0.0f)));

#if ARBITRARY_MESH_BOUNDING_VOLUME_CULLING
    if (!simpleBoxIntersectionTest(mesh.mesh.position_min, mesh.mesh.position_max, ro, rd)) {
        return -1;
    }
#endif

    float min_t = -1;

    size_t count = mesh.mesh.indicesBuffer.byteLength / sizeof(unsigned int);
    for (int i = 0; i < count; i += 3) {
        unsigned int i0 = getData<unsigned int>(i, mesh.mesh.indicesBuffer, bufferBytes, bufferOffsets);
        unsigned int i1 = getData<unsigned int>(i+1, mesh.mesh.indicesBuffer, bufferBytes, bufferOffsets);
        unsigned int i2 = getData<unsigned int>(i+2, mesh.mesh.indicesBuffer, bufferBytes, bufferOffsets);
        glm::vec3 v0 = getData<glm::vec3>(i0, mesh.mesh.positionsBuffer, bufferBytes, bufferOffsets);
        glm::vec3 v1 = getData<glm::vec3>(i1, mesh.mesh.positionsBuffer, bufferBytes, bufferOffsets);
        glm::vec3 v2 = getData<glm::vec3>(i2, mesh.mesh.positionsBuffer, bufferBytes, bufferOffsets);
        glm::vec3 baryPosition;
        bool doesIntersect = glm::intersectRayTriangle(ro, rd, v0, v1, v2, baryPosition);

        if (doesIntersect) {
            glm::vec3 objspaceIntersection = interpolateBarycentric(baryPosition, v0, v1, v2);
            glm::vec3 tmp_intersectionPoint = multiplyMV(mesh.transform, glm::vec4(objspaceIntersection, 1.f));

            float t = glm::length(r.origin - tmp_intersectionPoint);
            if (min_t > 0.0f && t > min_t) {
                // found previous closer intersection
                continue;
            }

            glm::vec3 n0 = getData<glm::vec3>(i0, mesh.mesh.normalsBuffer, bufferBytes, bufferOffsets);
            glm::vec3 n1 = getData<glm::vec3>(i1, mesh.mesh.normalsBuffer, bufferBytes, bufferOffsets);
            glm::vec3 n2 = getData<glm::vec3>(i2, mesh.mesh.normalsBuffer, bufferBytes, bufferOffsets);
             //TODO: use bary coordinates to find normal
            glm::vec3 objspaceNormal = interpolateBarycentric(baryPosition, n0, n1, n2);
            //glm::vec3 objspaceNormal = glm::normalize(glm::cross(v1 - v0, v2 - v1));
            if (glm::dot(ro - objspaceIntersection, objspaceNormal) < 0) { // if the point is on the other side of the triangle than the normal would suggest
                objspaceNormal *= -1;
            }

            /*for (int j = 0; j < mesh.mesh.numTexCoords; j++) {
                glm::vec2 t0 = getData<glm::vec2>(i0, mesh.mesh.textureCoordsBuffers[j], bufferBytes, bufferOffsets);
                glm::vec2 t1 = getData<glm::vec2>(i1, mesh.mesh.textureCoordsBuffers[j], bufferBytes, bufferOffsets);
                glm::vec2 t2 = getData<glm::vec2>(i2, mesh.mesh.textureCoordsBuffers[j], bufferBytes, bufferOffsets);

                texCoords[j] = interpolateBarycentric(baryPosition, t0, t1, t2);
            }*/

            min_t = t;
            intersectionPoint = tmp_intersectionPoint;
            normal = glm::normalize(multiplyMV(mesh.invTranspose, glm::vec4(objspaceNormal, 0.f)));
        }
    }
    return min_t;
}