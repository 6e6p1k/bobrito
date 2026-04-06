#include "../src/tensor.cuh"
#include "../src/bobr.cuh"

int main(void)
{
    int batch = 64;
    int in_dim = 784;
    int hidden = 256;
    int out_dim = 10;

    Tensor x({batch, in_dim}, true); x.fill(0.5);
    Tensor W1({in_dim, hidden}, true); W1.fill(0.01);
    Tensor b1({1, hidden}, true); b1.fill(0.0);
    Tensor W2({hidden, out_dim}, true); W2.fill(0.01);
    Tensor b2({1, out_dim}, true); b2.fill(0.0);

    for (int i = 0; i < 1; i++) {
        std::cout << "iter " << i << std::endl;
        auto h = (x.matmul(W1) + b1).relu();
        auto out = h.matmul(W2) + b2;
        auto loss = out.sum(0);
        loss.build_backward();
        loss.evaluate();
        loss.geval();
    }
    return 0;
}