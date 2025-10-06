#pragma once

#include <cuda_runtime.h>

#include "glm/glm.hpp"
#define MAX_PATHTRACE_TEXTURES 8

#include <string>
#include <vector>
#include <array>

#define BACKGROUND_COLOR (glm::vec3(0.0f))

enum GeomType
{
    SPHERE,
    CUBE,
    MESH,
};

struct Ray
{
    glm::vec3 origin;
    glm::vec3 direction;
};

struct FlatBufferView {
    int buffer;
    size_t byteOffset;
    size_t byteLength;
    size_t byteStride;
};

struct ImageData {
    int width;
    int height;
    int component;
    int bitDepthPerChannel;
    int pixelType;
};

struct Geom
{
    enum GeomType type;
    int materialid;
    glm::mat4 transform;
    glm::mat4 inverseTransform;
    glm::mat4 invTranspose;
    struct {
        int mode;
        glm::vec3 position_min;
        glm::vec3 position_max;
        FlatBufferView indicesBuffer;
        FlatBufferView positionsBuffer;
        FlatBufferView normalsBuffer;
        FlatBufferView textureCoordsBuffers[MAX_PATHTRACE_TEXTURES];
        int numTexCoords = 0;
    } mesh;
};

struct TextureImageInfo {
    bool exists = false;
    int imageBufferIdx;
    struct {
        int minFilter;
        int magFilter;
        int wrapS;
        int wrapT;
    } sampler;
    int texCoordsIdx = 0;
};

struct Material
{
    glm::vec3 color;
    struct
    {
        float exponent;
        glm::vec3 color;
    } specular;
    float hasReflective;
    float hasRefractive;
    float indexOfRefraction;
    glm::vec3 emittance;
    TextureImageInfo baseColorTexture;
};



struct Camera
{
    glm::ivec2 resolution;
    glm::vec3 position;
    glm::vec3 lookAt;
    glm::vec3 view;
    glm::vec3 up;
    glm::vec3 right;
    glm::vec2 fov;
    glm::vec2 pixelLength;
};

struct RenderState
{
    std::vector<Camera> cameras;
    int activeCamera = 0;
    unsigned int iterations;
    int traceDepth;
    std::vector<glm::vec3> image;
    std::string imageName;
};

struct PathSegment
{
    Ray ray;
    glm::vec3 color;
    glm::vec3 accumulatedColor;
    int pixelIndex;
    int remainingBounces;
};

// Use with a corresponding PathSegment to do:
// 1) color contribution computation
// 2) BSDF evaluation: generate a new ray
struct ShadeableIntersection
{
  int materialId;
  float t;
  glm::vec3 surfaceNormal;
  glm::vec2 texCoords0;
  glm::vec2 texCoords1;
  glm::vec2 texCoords2;
  glm::vec2 texCoords3;
  glm::vec2 texCoords4;
  glm::vec2 texCoords5;
  glm::vec2 texCoords6;
  glm::vec2 texCoords7;
  glm::vec2 texCoords8;
  glm::vec2 texCoords9;
  glm::vec2 texCoords10;
  glm::vec2 texCoords11;
  glm::vec2 texCoords12;
  glm::vec2 texCoords13;
  glm::vec2 texCoords14;
  glm::vec2 texCoords15;
  //glm::vec2 texCoordsSI[MAX_PATHTRACE_TEXTURES];
  //std::array<glm::vec2, MAX_PATHTRACE_TEXTURES> texCoords;
};


// insane hack because using any type of array for texcoords in shadeableintersection caused launch failures on kernels
__host__ __device__ glm::vec2& getTexCoords(ShadeableIntersection& si, int idx);

__host__ __device__ const glm::vec2& getTexCoords(const ShadeableIntersection& si, int idx);