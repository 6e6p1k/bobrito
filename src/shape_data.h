#include "allocator.cuh"

// ShapeData
struct ShapeData
{
    int ndim = -1;
    int size = -1;
    int data[0];

    static ShapeData* init(int ndim);
    int* shape();
    int* strides();
    void print();
};

using ShapePtr = std::shared_ptr<ShapeData>;
ShapePtr make_shape_data(int ndim, int* shape, int* strides = nullptr);

using ContentPtr = std::shared_ptr<ContentData>;
