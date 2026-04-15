#pragma once
#include "nodes.h"

void export_dot_full(Node* start, const char* filename);
std::string ptr_str(void* ptr);