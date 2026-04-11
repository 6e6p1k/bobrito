#include "nodes.h"
#include <memory>


void backward(std::shared_ptr<Node> loss);
void eval(std::shared_ptr<Node> root);