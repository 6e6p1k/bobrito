#pragma once
#include <iostream>
#include <cmath>
#include <string>

static int _tests_passed = 0;
static int _tests_failed = 0;

#define ASSERT(cond, msg) \
    if (!(cond)) { throw std::runtime_error(msg); }

#define ASSERT_NEAR(a, b, tol) \
    if (std::fabs((a) - (b)) > (tol)) { \
        throw std::runtime_error( \
            "Expected " + std::to_string(a) + " ≈ " + std::to_string(b) + \
            " (tol=" + std::to_string(tol) + ")"); \
    }

#define TEST(name) \
    void name(); \
    struct name##_reg { name##_reg() { \
        std::cout << "  " #name "... "; \
        try { name(); _tests_passed++; std::cout << "PASS" << std::endl; } \
        catch (const std::exception& e) { _tests_failed++; std::cout << "FAIL: " << e.what() << std::endl; } \
    } } name##_inst; \
    void name()

#define TEST_SUMMARY() \
    std::cout << "\n" << _tests_passed << " passed, " << _tests_failed << " failed\n"; \
    return _tests_failed > 0 ? 1 : 0;