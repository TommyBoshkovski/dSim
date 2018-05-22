extern "C"
{


void checkCUDA();

void allocateArray(void **devPtr, int size);
void freeArray(void *devPtr);

void threadSync();

void copyArrayFromDevice(void* host, const void* device, unsigned int vbo, int size);
void copyArrayToDevice(void* device, const void* host, int offset, int size);
void registerGLBufferObject(unsigned int vbo);
void unregisterGLBufferObject(unsigned int vbo);
void copyConstantToDevice(void* device, const void* host, int offset, int size);

void bindCubeCounter(uint* ptr, int size);
void unbindCubeCounter();
void bindTrianglesInCubes(uint* ptr, int size);
void unbindTrianglesInCubes();
//void bindTrgls(uint* ptr, int size);
//void unbindTrgls();
void bindVertices(float* ptr, int size);
void unbindVertices();
void bindTriangleHelpers(float* ptr, int size);
void unbindTriangleHelpers();
void bindRTreeArray(float* ptr, int size);
void unbindRTreeArray();
void bindTreeIndexArray(uint* ptr, int size);
void unbindTreeIndexArray();
void bindTriInfo(uint* ptr, int size);
void unbindTriInfo();

void integrateSystem(
			uint vboPos,
			uint* randSeed,
			spinData* spinInfo,
			float deltaTime,
			float permeability,
			uint numBodies,
			float3 gradient,
			float phaseConstant,
			uint iterations, uint* trianglesInCubes, uint* cubeCounter,uint m_nMembraneTypes,
uint m_nPosValues, uint m_numSpins, uint m_nSeedValues, uint m_numCompartments, float* m_hT2Values, float* m_hStdDevs, uint m_reflectionType, uint m_triSearchMethod, uint m_nFibers
			);

void integrateSystemVBO(
			float* vboPos,
			uint* randSeed,
			spinData* spinInfo,
			float deltaTime,
			float permeability,
			uint numBodies,
			float3 gradient,
			float phaseConstant,
			uint iterations, uint* trianglesInCubes, uint* cubeCounter,uint m_nMembraneTypes,
uint m_nPosValues, uint m_numSpins, uint m_nSeedValues, uint m_numCompartments, float* m_hT2Values, float* m_hStdDevs, uint m_reflectionType, uint m_triSearchMethod, uint m_nFibers, uint m_nSpinValues, uint m_totalNumCubes, uint m_maxTrianglesPerCube,uint m_numCubes, uint m_posVbo
			);
}
