#pragma once 
#include "nodes.h"
#include <memory>


struct Primitive {
    virtual void eval(Node* node) = 0;
    virtual void vjp(std::shared_ptr<Node> fwd,
                     std::shared_ptr<Node> dC,
                     std::shared_ptr<Node> out[]) {}
    virtual const char* name() = 0;
    virtual const char* symbol() = 0;
    virtual ~Primitive() = default;
};
