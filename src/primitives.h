#pragma once

#include <memory>
#include <string>
#include "nodes.h"

enum class PrimCategory {
    Leaf,
    Source,       // full, arange, uniform, normal
    EwiseUnary,   // relu, exp, log, neg, sqrt
    EwiseBinary,  // add, sub, mul, div, pow
    ReduceLD,     // sum_ld, min_ld, max_ld, sum, min, max
    Movement,     // reshape, transpose, contiguous, expand_ld
    MatMul,
    Skip,         // reduce_to_shape
    Special,      // softmax, max_min_bwd, bwd_relu, bwd_div, bwd_pow
    Identity
};

struct Primitive {
    virtual void eval(Node* node) = 0;
    virtual void vjp(std::shared_ptr<Node> fwd, std::shared_ptr<Node> dC,
                     std::shared_ptr<Node> out[]) {}
    virtual void allocate(Node* node) {
        if (!node->is_data()) node->d_data = std::make_shared<ContentData>();
        if (!node->is_data_allocated()) node->d_data->allocate(node->get_size());
    }
    virtual const std::string name() = 0;
    virtual const std::string symbol() = 0;
    virtual PrimCategory category() const = 0;
    virtual ~Primitive() = default;
};

// ========================= LEAF ==========================
struct LeafPrim : Primitive {
    void eval(Node* node) override {}
    const std::string name() override { return "Leaf"; }
    const std::string symbol() override { return "leaf"; }
    PrimCategory category() const override { return PrimCategory::Leaf; }
};
extern LeafPrim prim_leaf;

// ======================= IDENTITY ========================
struct IdentityPrim : Primitive {
    void eval(Node* node) override {}
    void allocate(Node* node) override;
    const std::string name() override { return "Identity"; }
    const std::string symbol() override { return "id"; }
    PrimCategory category() const override { return PrimCategory::Identity; }
};
extern IdentityPrim prim_identity;

// ========================= RELU ==========================
struct ReluPrim : Primitive {
    void eval(Node* node) override;
    void vjp(std::shared_ptr<Node> fwd, std::shared_ptr<Node> grad,
             std::shared_ptr<Node> out[]) override;
    const std::string name() override { return "ReLU"; }
    const std::string symbol() override { return "relu"; }
    PrimCategory category() const override { return PrimCategory::EwiseUnary; }
};
extern ReluPrim prim_relu;

// ======================= BWD RELU ========================
struct BwdReluPrim : Primitive {
    void eval(Node* node) override;
    const std::string name() override { return "BwdReLU"; }
    const std::string symbol() override { return "bwd_relu"; }
    PrimCategory category() const override { return PrimCategory::Special; }
};
extern BwdReluPrim prim_bwd_relu;

// ========================== NEG ==========================
struct NegPrim : Primitive {
    void eval(Node* node) override;
    void vjp(std::shared_ptr<Node> fwd, std::shared_ptr<Node> bwd,
             std::shared_ptr<Node> out[]) override;
    const std::string name() override { return "Neg"; }
    const std::string symbol() override { return "-"; }
    PrimCategory category() const override { return PrimCategory::EwiseUnary; }
};
extern NegPrim prim_neg;

// ========================== EXP ==========================
struct ExpPrim : Primitive {
    void eval(Node* node) override;
    void vjp(std::shared_ptr<Node> fwd, std::shared_ptr<Node> bwd,
             std::shared_ptr<Node> out[]) override;
    const std::string name() override { return "Exp"; }
    const std::string symbol() override { return "exp"; }
    PrimCategory category() const override { return PrimCategory::EwiseUnary; }
};
extern ExpPrim prim_exp;

// ========================= SQRT ==========================
struct SqrtPrim : Primitive {
    void eval(Node* node) override;
    const std::string name() override { return "Sqrt"; }
    const std::string symbol() override { return "sqrt"; }
    PrimCategory category() const override { return PrimCategory::EwiseUnary; }
};
extern SqrtPrim prim_sqrt;

// ========================== LOG ==========================
struct LogPrim : Primitive {
    void eval(Node* node) override;
    void vjp(std::shared_ptr<Node> fwd, std::shared_ptr<Node> bwd,
             std::shared_ptr<Node> out[]) override;
    const std::string name() override { return "Log"; }
    const std::string symbol() override { return "log"; }
    PrimCategory category() const override { return PrimCategory::EwiseUnary; }
};
extern LogPrim prim_log;

// ========================= FULL ==========================
struct FullPrim : Primitive {
    void eval(Node* node) override;
    const std::string name() override { return "Full"; }
    const std::string symbol() override { return "full"; }
    PrimCategory category() const override { return PrimCategory::Source; }
};
extern FullPrim prim_full;

// ======================== ARANGE =========================
struct ArangePrim : Primitive {
    void eval(Node* node) override;
    const std::string name() override { return "Arange"; }
    const std::string symbol() override { return "arange"; }
    PrimCategory category() const override { return PrimCategory::Source; }
};
extern ArangePrim prim_arange;

// ======================== UNIFORM ========================
struct UniformPrim : Primitive {
    void eval(Node* node) override;
    const std::string name() override { return "Uniform"; }
    const std::string symbol() override { return "uniform"; }
    PrimCategory category() const override { return PrimCategory::Source; }
};
extern UniformPrim prim_uniform;

// ======================== NORMAL =========================
struct NormalPrim : Primitive {
    void eval(Node* node) override;
    const std::string name() override { return "Normal"; }
    const std::string symbol() override { return "normal"; }
    PrimCategory category() const override { return PrimCategory::Source; }
};
extern NormalPrim prim_normal;

// ========================== ADD ==========================
struct AddPrim : Primitive {
    void eval(Node* node) override;
    void vjp(std::shared_ptr<Node> fwd, std::shared_ptr<Node> bwd,
             std::shared_ptr<Node> out[]) override;
    const std::string name() override { return "Add"; }
    const std::string symbol() override { return "+"; }
    PrimCategory category() const override { return PrimCategory::EwiseBinary; }
};
extern AddPrim prim_add;

// ========================== SUB ==========================
struct SubPrim : Primitive {
    void eval(Node* node) override;
    void vjp(std::shared_ptr<Node> fwd, std::shared_ptr<Node> bwd,
             std::shared_ptr<Node> out[]) override;
    const std::string name() override { return "Sub"; }
    const std::string symbol() override { return "-"; }
    PrimCategory category() const override { return PrimCategory::EwiseBinary; }
};
extern SubPrim prim_sub;

// ========================== MUL ==========================
struct MulPrim : Primitive {
    void eval(Node* node) override;
    void vjp(std::shared_ptr<Node> fwd, std::shared_ptr<Node> bwd,
             std::shared_ptr<Node> out[]) override;
    const std::string name() override { return "Mul"; }
    const std::string symbol() override { return "*"; }
    PrimCategory category() const override { return PrimCategory::EwiseBinary; }
};
extern MulPrim prim_mul;

// ========================== DIV ==========================
struct DivPrim : Primitive {
    void eval(Node* node) override;
    void vjp(std::shared_ptr<Node> fwd, std::shared_ptr<Node> bwd,
             std::shared_ptr<Node> out[]) override;
    const std::string name() override { return "Div"; }
    const std::string symbol() override { return "/"; }
    PrimCategory category() const override { return PrimCategory::EwiseBinary; }
};
extern DivPrim prim_div;

// ======================== BWD DIV ========================
struct BwdDivPrim : Primitive {
    void eval(Node* node) override;
    const std::string name() override { return "BwdDiv"; }
    const std::string symbol() override { return "bwd_div"; }
    PrimCategory category() const override { return PrimCategory::Special; }
};
extern BwdDivPrim prim_bwd_div;

// ========================== POW ==========================
struct PowPrim : Primitive {
    void eval(Node* node) override;
    void vjp(std::shared_ptr<Node> fwd, std::shared_ptr<Node> bwd,
             std::shared_ptr<Node> out[]) override;
    const std::string name() override { return "Pow"; }
    const std::string symbol() override { return "^"; }
    PrimCategory category() const override { return PrimCategory::EwiseBinary; }
};
extern PowPrim prim_pow;

// ======================== BWD POW ========================
struct BwdPowPrim : Primitive {
    void eval(Node* node) override;
    const std::string name() override { return "BwdPow"; }
    const std::string symbol() override { return "bwd_pow"; }
    PrimCategory category() const override { return PrimCategory::Special; }
};
extern BwdPowPrim prim_bwd_pow;

// ======================== MATMUL =========================
struct MatMulPrim : Primitive {
    void eval(Node* node) override;
    void vjp(std::shared_ptr<Node> fwd, std::shared_ptr<Node> bwd,
             std::shared_ptr<Node> out[]) override;
    const std::string name() override { return "MatMul"; }
    const std::string symbol() override { return "@"; }
    PrimCategory category() const override { return PrimCategory::MatMul; }
};
extern MatMulPrim prim_matmul;

// ====================== MAX/MIN BWD ======================
struct MaxMinBwdPrim : Primitive {
    void eval(Node* node) override;
    const std::string name() override { return "MaxMinBwd"; }
    const std::string symbol() override { return "max_min_bwd()"; }
    PrimCategory category() const override { return PrimCategory::Special; }
};
extern MaxMinBwdPrim prim_max_min_bwd;

// ======================== SUM LD =========================
struct SumLDPrim : Primitive {
    void eval(Node* node) override;
    void vjp(std::shared_ptr<Node> fwd, std::shared_ptr<Node> bwd,
             std::shared_ptr<Node> out[]) override;
    const std::string name() override { return "SumLD"; }
    const std::string symbol() override { return "sum(dim=-1)"; }
    PrimCategory category() const override { return PrimCategory::ReduceLD; }
};
extern SumLDPrim prim_sum_ld;

// ======================== MIN LD =========================
struct MinLDPrim : Primitive {
    void eval(Node* node) override;
    void vjp(std::shared_ptr<Node> fwd, std::shared_ptr<Node> grad,
             std::shared_ptr<Node> out[]) override;
    const std::string name() override { return "MinLD"; }
    const std::string symbol() override { return "min(dim=-1)"; }
    PrimCategory category() const override { return PrimCategory::ReduceLD; }
};
extern MinLDPrim prim_min_ld;

// ======================== MAX LD =========================
struct MaxLDPrim : Primitive {
    void eval(Node* node) override;
    void vjp(std::shared_ptr<Node> fwd, std::shared_ptr<Node> grad,
             std::shared_ptr<Node> out[]) override;
    const std::string name() override { return "MaxLD"; }
    const std::string symbol() override { return "max(dim=-1)"; }
    PrimCategory category() const override { return PrimCategory::ReduceLD; }
};
extern MaxLDPrim prim_max_ld;

// ======================= EXPAND LD =======================
struct ExpandLDPrim : Primitive {
    void eval(Node* node) override;
    const std::string name() override { return "ExpandLD"; }
    const std::string symbol() override { return "expand(dim=-1)"; }
    PrimCategory category() const override { return PrimCategory::Movement; }
};
extern ExpandLDPrim prim_expand_ld;

// ========================== SUM ==========================
struct SumPrim : Primitive {
    void eval(Node* node) override;
    const std::string name() override { return "Sum"; }
    const std::string symbol() override { return "sum()"; }
    PrimCategory category() const override { return PrimCategory::ReduceLD; }
};
extern SumPrim prim_sum;

// ========================== MIN ==========================
struct MinPrim : Primitive {
    void eval(Node* node) override;
    const std::string name() override { return "Min"; }
    const std::string symbol() override { return "min()"; }
    PrimCategory category() const override { return PrimCategory::ReduceLD; }
};
extern MinPrim prim_min;

// ========================== MAX ==========================
struct MaxPrim : Primitive {
    void eval(Node* node) override;
    const std::string name() override { return "Max"; }
    const std::string symbol() override { return "max()"; }
    PrimCategory category() const override { return PrimCategory::ReduceLD; }
};
extern MaxPrim prim_max;

// ======================= TRANSPOSE =======================
struct TransposePrim : Primitive {
    void eval(Node* node) override;
    void vjp(std::shared_ptr<Node> fwd, std::shared_ptr<Node> bwd,
             std::shared_ptr<Node> out[]) override;
    const std::string name() override { return "Transpose"; }
    const std::string symbol() override { return "transpose"; }
    PrimCategory category() const override { return PrimCategory::Movement; }
};
extern TransposePrim prim_transpose;

// ====================== CONTIGUOUS =======================
struct ContiguousPrim : Primitive {
    void eval(Node* node) override;
    void vjp(std::shared_ptr<Node> fwd, std::shared_ptr<Node> bwd,
             std::shared_ptr<Node> out[]) override;
    const std::string name() override { return "Contiguous"; }
    const std::string symbol() override { return "contiguous"; }
    PrimCategory category() const override { return PrimCategory::Movement; }
};
extern ContiguousPrim prim_contiguous;

// ======================== RESHAPE ========================
struct ReshapePrim : Primitive {
    void eval(Node* node) override {}
    void vjp(std::shared_ptr<Node> fwd, std::shared_ptr<Node> bwd,
             std::shared_ptr<Node> out[]) override;
    void allocate(Node* node) override;
    const std::string name() override { return "Reshape"; }
    const std::string symbol() override { return "reshape"; }
    PrimCategory category() const override { return PrimCategory::Movement; }
};
extern ReshapePrim prim_reshape;

// ==================== REDUCE TO SHAPE ====================
struct ReduceToShapePrim : Primitive {
    void eval(Node* node) override {}
    const std::string name() override { return "Skip"; }
    const std::string symbol() override { return "skip"; }
    PrimCategory category() const override { return PrimCategory::Skip; }
};
extern ReduceToShapePrim prim_reduce_to_shape;

// ======================== SOFTMAX ========================
struct SoftmaxPrim : Primitive {
    void eval(Node* node) override {}
    const std::string name() override { return "Softmax"; }
    const std::string symbol() override { return "softmax"; }
    PrimCategory category() const override { return PrimCategory::Special; }
};
extern SoftmaxPrim prim_softmax;

// // ========================== DIV ==========================
// struct DivPrim : Primitive {
//     void eval(Node* node) override;
//     void vjp(std::shared_ptr<Node> fwd, std::shared_ptr<Node> bwd,
//              std::shared_ptr<Node> out[]) override;
//     const std::string name() override { return "Div"; }
//     const std::string symbol() override { return "/"; }
//     PrimCategory category() const override { return PrimCategory::EwiseBinary; }
// };
// extern DivPrim prim_div;