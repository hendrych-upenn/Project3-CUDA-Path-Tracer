#pragma once

#include "sceneStructs.h"
#include "tiny_gltf.h"
#include <vector>
#include <optional>

class Scene
{
private:
    void loadFromJSON(const std::string& jsonName);
    void loadFromGLTF(const std::string& gltfName);
public:
    Scene(std::string filename);

    std::vector<Geom> geoms;
    std::vector<Material> materials;
    RenderState state;
    std::optional<tinygltf::Model> model;
    std::vector<uint8_t> bufferBytes;
    std::vector<size_t> bufferOffsets;
    std::vector<uint8_t> imageBufferBytes;
    std::vector<size_t> imageBufferOffsets;
    std::vector<ImageData> imagesData;
};
