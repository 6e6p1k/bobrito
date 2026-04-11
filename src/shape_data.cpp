#include <iostream>
#include <memory>

#include "shape_data.h"
#include "utils.h"


using namespace std;

int* ShapeData::shape() { return &data[0]; }
int* ShapeData::strides() { return &data[ndim]; }

ShapeData* ShapeData::init(int ndim)
{   
    size_t total_bytes = sizeof(ShapeData) + sizeof(int) * ndim * 2;

    ShapeData* sd = (ShapeData*)malloc(total_bytes);
    if (sd) sd->ndim = ndim;

    return sd;
}

shared_ptr<ShapeData> make_shape_data(int ndim, int* shape, int* strides)
{
    ShapeData* sd = ShapeData::init(ndim);
    sd->size = utils::numel(ndim, shape);
    // case 1: strides are not specified
    if (strides == nullptr)
    {
        utils::compute_strides(sd->strides(), ndim, shape);
        for (int i = 0; i < ndim; i++) sd->shape()[i] = shape[i];
        return shared_ptr<ShapeData>(sd, ::free);
    }
    // case 2: strides are specified
    for (int i = 0; i < ndim; i++)
    {
        sd->shape()[i] = shape[i];
        sd->strides()[i] = strides[i];
    }
    return shared_ptr<ShapeData>(sd, ::free);
}