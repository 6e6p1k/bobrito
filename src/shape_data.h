#pragma once
#include <iostream>
#include <memory>

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

std::shared_ptr<ShapeData> make_shape_data(int ndim, int* shape, int* strides = nullptr);

