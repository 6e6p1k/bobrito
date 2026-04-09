#include <memory>
#include <vector>
#include "node.h"

namespace layers

{

enum struct Layers {Linear};

struct layer 
{
    std::vector<std::shared_ptr<Node>> parameters;

    virtual void apply(std::shared_ptr<Node> node);
    virtual void bwd(void);
};

struct linear_layer : layer
{
    void apply(void);
};





}