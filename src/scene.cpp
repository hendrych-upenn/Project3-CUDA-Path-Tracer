#include "scene.h"

#include "utilities.h"

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/quaternion.hpp>
#include <glm/gtx/string_cast.hpp>
#include "json.hpp"

#include "tiny_gltf.h"

#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>

using namespace std;
using json = nlohmann::json;

Scene::Scene(string filename)
{
    cout << "Reading scene from " << filename << " ..." << endl;
    cout << " " << endl;
    auto ext = filename.substr(filename.find_last_of('.'));
    if (ext == ".json")
    {
        loadFromJSON(filename);
        return;
    }
    else if (ext == ".gltf" || ext == ".glb") {
        loadFromGLTF(filename);
    }
    else
    {
        cout << "Couldn't read from " << filename << endl;
        exit(-1);
    }
}

FlatBufferView flatBufferViewFromAccessor(const tinygltf::Model& model, const tinygltf::Accessor& accessor) {
    tinygltf::BufferView bufferView = model.bufferViews[accessor.bufferView];
    return {
        bufferView.buffer, //buffer
        accessor.byteOffset + bufferView.byteOffset,//offset
        bufferView.byteLength, //length
        bufferView.byteStride, //stride
    };

}

bool geomFromPrimitive(Geom& geom, const tinygltf::Model& model, const tinygltf::Primitive& primitive) {

    if (primitive.mode != TINYGLTF_MODE_TRIANGLES) {
        cout << "Mesh mode not supported, skipping..." << endl;
        return false;
    }
    geom.type = MESH;
    geom.materialid = primitive.material;
    geom.transform = glm::mat4();
    geom.inverseTransform = glm::mat4();
    geom.invTranspose = glm::mat4();
    geom.mesh.mode = primitive.mode;

    if (primitive.indices == -1) {
        cout << "Mesh indices not found, skipping..." << endl;
        return false;
    }
    tinygltf::Accessor indicesAccessor = model.accessors[primitive.indices];
    cout << "Indices has component type number: " << indicesAccessor.componentType << "\n" << "Unsigned int: " << TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT << endl;
    geom.mesh.indicesBuffer = flatBufferViewFromAccessor(model, indicesAccessor);

    const auto& attributes = primitive.attributes;

    auto positionAttribute = attributes.find("POSITION");
    if (positionAttribute == attributes.end()) {
        cout << "Mesh POSITION attribute not found, skipping..." << endl;
        return false;
    }
    int positionAccessorIdx = (*positionAttribute).second;
    tinygltf::Accessor positionAccessor = model.accessors[positionAccessorIdx];
    if (positionAccessor.componentType != TINYGLTF_COMPONENT_TYPE_FLOAT) {
        cout << "Mesh POSITION accessor component type not float, skipping..." << endl;
        return false;
    }
    if (positionAccessor.type != TINYGLTF_TYPE_VEC3) {
        cout << "Mesh POSITION accessor type not vec3, skipping..." << endl;
        return false;
    }
    geom.mesh.positionsBuffer = flatBufferViewFromAccessor(model, positionAccessor);

    geom.mesh.position_min = glm::vec3(glm::make_vec3(positionAccessor.minValues.data()));
    geom.mesh.position_max = glm::vec3(glm::make_vec3(positionAccessor.maxValues.data()));

    auto normalAttribute = attributes.find("NORMAL");
    if (normalAttribute == attributes.end()) {
        cout << "Mesh NORMAL attribute not found, skipping..." << endl;
        return false;
    }
    int normalAccessorIdx = (*normalAttribute).second;
    tinygltf::Accessor normalAccessor = model.accessors[normalAccessorIdx];
    if (normalAccessor.componentType != TINYGLTF_COMPONENT_TYPE_FLOAT) {
        cout << "Mesh NORMAL accessor component type not float, skipping..." << endl;
        return false;
    }
    if (normalAccessor.type != TINYGLTF_TYPE_VEC3) {
        cout << "Mesh NORMAL accessor type not vec3, skipping..." << endl;
        return false;
    }
    geom.mesh.normalsBuffer = flatBufferViewFromAccessor(model, normalAccessor);

    /*geom.mesh.numTexCoords = 0;
    for (int i = 0; attributes.find("TEXCOORD_" + i) != attributes.end(); i++) {
        int texcoordAccessorIdx = (*attributes.find("TEXCOORD_" + i)).second;
        tinygltf::Accessor texcoordAccessor = model.accessors[texcoordAccessorIdx];
        if (texcoordAccessor.type != TINYGLTF_TYPE_VEC2) {
            cout << "Mesh TEXCOORD_" << i << " accessor type not vec2, skipping..." << endl;
            return false;
        }
        if (texcoordAccessor.componentType != TINYGLTF_COMPONENT_TYPE_FLOAT) {
            cout << "Mesh TEXCOORD_" << i << " accessor component type not float, skipping..." << endl;
            return false;
        }
        geom.mesh.textureCoordsBuffers[i] = flatBufferViewFromAccessor(model, texcoordAccessor);
        geom.mesh.numTexCoords++;
    }*/


    return true;
}

void getGeometriesFromNodes(std::vector<Geom>& geoms, std::vector<Camera> cameras, const tinygltf::Model& model, const std::vector<int>& nodes) {
    for (int node_idx : nodes) {
        const tinygltf::Node& node = model.nodes[node_idx];
        glm::mat4 transformation = glm::mat4();
        if (node.matrix.size() != 0) {
            transformation = glm::make_mat4(node.matrix.data());
        }
        else if (node.translation.size() != 0 || node.rotation.size() != 0 || node.scale.size() != 0) {
            if (node.translation.size() != 0) {
                transformation = glm::translate(transformation, glm::vec3(glm::make_vec3(node.translation.data())));
            }
            if (node.rotation.size() != 0) {
                transformation *= glm::toMat4(glm::quat(glm::make_quat(node.rotation.data())));
            }
            if (node.scale.size() != 0) {
                transformation = glm::scale(transformation, glm::vec3(glm::make_vec3(node.scale.data())));
            }
        }
        std::vector<Geom> node_geoms;
        if (node.mesh != -1) {
            const tinygltf::Mesh& mesh = model.meshes[node.mesh];
            for (const tinygltf::Primitive& primitive : mesh.primitives) {

                Geom geom;

                bool ret = geomFromPrimitive(geom, model, primitive);

                if (!ret) {
                    cout << "Skipping..." << endl;
                    continue;
                }

                node_geoms.emplace_back(geom);
            }
        }
        std::vector<Camera> node_cameras{};
        getGeometriesFromNodes(node_geoms, node_cameras, model, node.children);
        for (Geom& geom : node_geoms) {
            geom.transform = transformation * geom.transform;
            geom.inverseTransform = glm::inverse(geom.transform);
            geom.invTranspose = glm::inverseTranspose(geom.transform);
        }
        geoms.insert(geoms.end(), node_geoms.begin(), node_geoms.end());
        cameras.insert(cameras.end(), node_cameras.begin(), node_cameras.end());
    }
    return;
}

void loadMaterials(std::vector<Material>& materials, const tinygltf::Model& model) 
{
    for (const tinygltf::Material& material : model.materials) {
        Material mat;
        
        mat.color = glm::vec3(glm::make_vec3(material.pbrMetallicRoughness.baseColorFactor.data()));
        mat.emittance = glm::vec3(glm::make_vec3(material.emissiveFactor.data()));
        
        /*if (material.pbrMetallicRoughness.baseColorTexture.index != -1) {
            const auto& colorTexture = material.pbrMetallicRoughness.baseColorTexture;
            mat.gltf.baseColorTexture.exists = true;
            mat.gltf.baseColorTexture.texCoordsIdx = colorTexture.texCoord;

            int textureIndex = colorTexture.index;
            const tinygltf::Texture& texture = model.textures[textureIndex];
            mat.gltf.baseColorTexture.imageBufferIdx = texture.source;

            const tinygltf::Sampler sampler = model.samplers[texture.sampler];
            auto& matSampler = mat.gltf.baseColorTexture.sampler;

            matSampler.magFilter = sampler.magFilter;
            matSampler.minFilter = sampler.minFilter;
            matSampler.wrapS = sampler.wrapS;
            matSampler.wrapT = sampler.wrapT;
        }*/
        
        materials.emplace_back(mat);
    }
}

Camera getDefaultCamera() {
    Camera camera;
    camera.resolution.x = 800;
    camera.resolution.y = 800;
    float fovy = 45.0;
    camera.position = glm::vec3(0.0f, 1.0f, 2.0f);
    camera.lookAt = glm::vec3(0.0f, 0.0f, 0.0f);
    camera.up = glm::vec3(0.0f, 1.0f, 0.0f);

    //calculate fov based on resolution
    float yscaled = tan(fovy * (PI / 180));
    float xscaled = (yscaled * camera.resolution.x) / camera.resolution.y;
    float fovx = (atan(xscaled) * 180) / PI;
    camera.fov = glm::vec2(fovx, fovy);

    camera.view = glm::normalize(camera.lookAt - camera.position);
    camera.right = glm::normalize(glm::cross(camera.view, camera.up));
    camera.pixelLength = glm::vec2(2 * xscaled / (float)camera.resolution.x,
        2 * yscaled / (float)camera.resolution.y);
    return camera;
}

void getBufferBytes(std::vector<uint8_t>& bytes, std::vector<size_t>& offsets, const tinygltf::Model& model) {
    const std::vector<tinygltf::Buffer>& buffers = model.buffers;
    for (size_t i = 0; i < buffers.size(); i++) {
        const tinygltf::Buffer& buffer = buffers[i];
        offsets.push_back(bytes.size());
        bytes.insert(bytes.end(), buffer.data.begin(), buffer.data.end());;
    }
}

void getImageBuffers(std::vector<uint8_t>& bytes, std::vector<size_t>& offsets, std::vector<ImageData>& imagesData, const tinygltf::Model& model) {
    const std::vector<tinygltf::Image>& images = model.images;
    for (size_t i = 0; i < images.size(); i++) {
        const tinygltf::Image& image = images[i];
        offsets.push_back(bytes.size());
        bytes.insert(bytes.end(), image.image.begin(), image.image.end());;
        ImageData imageData;
        imageData.width = image.width;
        imageData.height = image.height;
        imageData.component = image.component;
        imageData.bitDepthPerChannel = image.bits;
        imageData.pixelType = image.pixel_type;
        imagesData.emplace_back(std::move(imageData));
    }
}

void Scene::loadFromGLTF(const std::string& gltfName) {
    tinygltf::Model model;
    tinygltf::TinyGLTF loader;
    std::string err;
    std::string warn;

    bool ret = false;
    auto ext = gltfName.substr(gltfName.find_last_of('.'));
    if (ext == ".gltf")
    {
        ret = loader.LoadASCIIFromFile(&model, &err, &warn, gltfName);
    }
    else {

        ret = loader.LoadBinaryFromFile(&model, &err, &warn, gltfName);
    }
    if (!warn.empty()) {
        cout << "GLTF Warning: " << warn << endl;
    }
    if (!err.empty()) {
        cout << "GLTF Error: " << err << endl;
    }
    if (!ret) {
        cout << "Couldn't load gltf from " << gltfName << endl;
        exit(-1);
    }

    tinygltf::Scene scene;
    if (model.defaultScene == -1) {
        scene = model.scenes[0];
    }
    else {
        scene = model.scenes[model.defaultScene];
    }

    std::vector<Geom> geoms;
    std::vector<Camera> cameras;

    getGeometriesFromNodes(geoms, cameras, model, scene.nodes);

    std::vector<Material> materials;

    loadMaterials(materials, model);

    if (cameras.size() == 0) {
        cameras.push_back(getDefaultCamera());
    }

    getBufferBytes(this->bufferBytes, this->bufferOffsets, model);

    //getImageBuffers(this->imageBufferBytes, this->imageBufferOffsets, this->imagesData, model);

    this->geoms = std::move(geoms);
    this->materials = std::move(materials);
    this->model = std::move(model);
    RenderState& renderState = this->state;
    renderState.cameras = std::move(cameras);
    renderState.traceDepth = 8;
    renderState.iterations = 5000;
    state.imageName = gltfName;
    Camera& camera = renderState.cameras[renderState.activeCamera];
    int arrayLen = camera.resolution.x * camera.resolution.y;
    renderState.image.resize(arrayLen);
    std::fill(renderState.image.begin(), renderState.image.end(), glm::vec3());

}

void Scene::loadFromJSON(const std::string& jsonName)
{
    std::ifstream f(jsonName);
    json data = json::parse(f);
    const auto& materialsData = data["Materials"];
    std::unordered_map<std::string, uint32_t> MatNameToID;
    for (const auto& item : materialsData.items())
    {
        const auto& name = item.key();
        const auto& p = item.value();
        Material newMaterial{};
        // TODO: handle materials loading differently
        if (p["TYPE"] == "Diffuse")
        {
            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
        }
        else if (p["TYPE"] == "Emitting")
        {
            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
            newMaterial.emittance = glm::vec3((float)p["EMITTANCE"]);
        }
        else if (p["TYPE"] == "Specular")
        {
            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
        }
        MatNameToID[name] = materials.size();
        materials.emplace_back(newMaterial);
    }
    const auto& objectsData = data["Objects"];
    for (const auto& p : objectsData)
    {
        const auto& type = p["TYPE"];
        Geom newGeom;
        if (type == "cube")
        {
            newGeom.type = CUBE;
        }
        else
        {
            newGeom.type = SPHERE;
        }
        newGeom.materialid = MatNameToID[p["MATERIAL"]];
        const auto& trans_json = p["TRANS"];
        const auto& rotat_json = p["ROTAT"];
        const auto& scale_json = p["SCALE"];
        glm::vec3 translation = glm::vec3(trans_json[0], trans_json[1], trans_json[2]);
        glm::vec3 rotation = glm::vec3(rotat_json[0], rotat_json[1], rotat_json[2]);
        glm::vec3 scale = glm::vec3(scale_json[0], scale_json[1], scale_json[2]);
        newGeom.transform = utilityCore::buildTransformationMatrix(
            translation, rotation, scale);
        newGeom.inverseTransform = glm::inverse(newGeom.transform);
        newGeom.invTranspose = glm::inverseTranspose(newGeom.transform);

        geoms.push_back(newGeom);
    }
    const auto& cameraData = data["Camera"];
    this->model = std::nullopt;
    RenderState& state = this->state;
    state.cameras.emplace_back();
    Camera& camera = state.cameras[0];
    camera.resolution.x = cameraData["RES"][0];
    camera.resolution.y = cameraData["RES"][1];
    float fovy = cameraData["FOVY"];
    state.iterations = cameraData["ITERATIONS"];
    state.traceDepth = cameraData["DEPTH"];
    state.imageName = cameraData["FILE"];
    const auto& pos = cameraData["EYE"];
    const auto& lookat = cameraData["LOOKAT"];
    const auto& up = cameraData["UP"];
    camera.position = glm::vec3(pos[0], pos[1], pos[2]);
    camera.lookAt = glm::vec3(lookat[0], lookat[1], lookat[2]);
    camera.up = glm::vec3(up[0], up[1], up[2]);

    //calculate fov based on resolution
    float yscaled = tan(fovy * (PI / 180));
    float xscaled = (yscaled * camera.resolution.x) / camera.resolution.y;
    float fovx = (atan(xscaled) * 180) / PI;
    camera.fov = glm::vec2(fovx, fovy);

    camera.view = glm::normalize(camera.lookAt - camera.position);
    camera.right = glm::normalize(glm::cross(camera.view, camera.up));
    camera.pixelLength = glm::vec2(2 * xscaled / (float)camera.resolution.x,
        2 * yscaled / (float)camera.resolution.y);


    //set up render camera stuff
    int arraylen = camera.resolution.x * camera.resolution.y;
    state.image.resize(arraylen);
    std::fill(state.image.begin(), state.image.end(), glm::vec3());
}
