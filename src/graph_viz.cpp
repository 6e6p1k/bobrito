#include "graph_viz.h"
#include "nodes.h"
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>
#include <memory>

using namespace std;

static string escape_html(const string& s) {
    string out;
    for (char c : s) {
        if      (c == '<') out += "&lt;";
        else if (c == '>') out += "&gt;";
        else if (c == '&') out += "&amp;";
        else out += c;
    }
    return out;
}

static string shape_str(Node* node) {
    if (!node->is_meta()) return "[]";
    ostringstream ss;
    ss << "[";
    int ndim = node->get_ndim();
    int* shape = node->get_shape();
    for (int i = 0; i < ndim; i++) {
        if (i) ss << ", ";
        ss << shape[i];
    }
    ss << "]";
    return ss.str();
}

static string ptr_str(void* ptr) {
    ostringstream ss;
    ss << ptr;
    return ss.str();
}

static string make_row(const string& label, const string& value, const string& value_color = "#ffffff", const string& label_color = "#ffffff", const string& font_face = "iA Writer Mono S", int font_size = 14) {
    return "<TR><TD ALIGN=\"LEFT\">"
           "<FONT FACE=\"iA Writer Mono S\" POINT-SIZE=\"14\" COLOR=\"" + label_color + "\">" + label + "</FONT>"
           "<FONT FACE=\"iA Writer Mono S\" POINT-SIZE=\"14\" COLOR=\"" + value_color + "\">" + value + "</FONT>"
           "</TD></TR>";
}

static const char* op_color(const string& name) {
    if (name == "Add")    return "#4FC3F7";
    if (name == "Sub")    return "#80DEEA";
    if (name == "Mul")    return "#CE93D8";
    if (name == "Matmul") return "#90CAF9";
    if (name == "Neg")    return "#F48FB1";
    if (name == "Exp")    return "#FFE082";
    if (name == "Log")    return "#A5D6A7";
    if (name == "Relu")   return "#FFCC80";
    if (name == "Sum_ld") return "#FFD54F";
    if (name == "Div")    return "#B0BEC5";
    if (name == "Pow")    return "#BCAAA4";
    if (name == "Skip")   return "#B3E5FC";
    return "#EEEEEE";
}


static void dot_emit_node_label(ofstream& out, Node* node) {
    string id = "n" + to_string((uintptr_t)node);
    string shape_s = shape_str(node);

    bool scalar = shape_s == "[]";

    string name = scalar ? "Scalar" : node->prim->name();
    const char* color = op_color(name);

    string shape_row;
    if (!scalar)
        shape_row = make_row("Shape: ", shape_s, "#04ff00");
    else
        shape_row = make_row("Value: ", to_string((int)static_cast<ScalarNode*>(node)->value), "#04ff00");

    string node_ptr = make_row("Node ptr: ", ptr_str(node), "#00ffee");

    string label_row;
    if (!node->label.empty())
        label_row = make_row("Label: ", node->label, "#ffff00");

    string transposed_row;
    if (node->transposed)
        transposed_row = make_row("transposed = ", "true", "#ff00c8");
    else
        transposed_row = "";

    string requires_grad_row;
    if (node->requires_grad)
        requires_grad_row = make_row("requires_grad = ", "true", "#ff00c8");
    else
        requires_grad_row = "";


    string data_ptr_row;
    if (node->is_data_allocated())
        data_ptr_row = make_row("Data ptr: ", ptr_str(node->get_data()), "#00ffee");
    else data_ptr_row = "";

    string is_grad_buffer_row;
    string is_grad_buffer_ptr_row;
    if (node->is_grad_buffer()) {
        is_grad_buffer_row = make_row("Grad buffer ptr: ", ptr_str(node->grad_buffer.get()), "#00ffee");
        if (node->grad_buffer->is_data_allocated()) {
            is_grad_buffer_ptr_row = make_row("Grad buffer data ptr: ", ptr_str(node->grad_buffer->get_data()), "#00ffee");
        } else is_grad_buffer_ptr_row = make_row("Grad buffer data: ", "not allocated", "#ff5e5e");
    } else {
        is_grad_buffer_row = "";
        is_grad_buffer_ptr_row = "";
    }

    string is_owned_grad_row;
    string is_owned_grad_ptr_row;
    if (node->owned_grad) {
        is_owned_grad_row = make_row("Owned grad ptr: ", ptr_str(node->owned_grad.get()), "#00ffee");
        if (node->owned_grad->is_data_allocated()) {
            is_owned_grad_ptr_row = make_row("Owned grad data ptr: ", ptr_str(node->owned_grad->get_data()), "#00ffee");
        } else is_owned_grad_ptr_row = make_row("Owned grad data: ", "not allocated", "#ff5e5e");
    } else {
        is_owned_grad_row = "";
        is_owned_grad_ptr_row = "";
    }

    out << "    " << id << " [\n"
        << "        label=<"
        << "<TABLE BORDER=\"0\" CELLBORDER=\"0\" CELLSPACING=\"4\">"
        << "<TR><TD COLSPAN=\"2\">"
        // << "<FONT POINT-SIZE=\"20\" COLOR=\"#ffffff\">" << sym << " </FONT>"
        << "<FONT POINT-SIZE=\"18\" COLOR=\"#ffffff\"><B>" << name << "</B></FONT>"
        << "</TD></TR>"
        << label_row
        << shape_row
        << requires_grad_row
        << transposed_row
        << node_ptr
        << data_ptr_row
        << is_grad_buffer_row
        << is_grad_buffer_ptr_row
        << is_owned_grad_row
        << is_owned_grad_ptr_row;
    

    out << "</TABLE>>\n"
        << "        shape=box style=\"rounded\"\n"
        << "        penwidth=2 color=\"#FFFFFF\"\n"
        << "    ];\n";
}

static void dot_emit_node_full(ofstream& out, Node* node,
                               unordered_set<Node*>& visited_nodes,
                               unordered_set<Node*>& visited_edges) {
    if (!visited_nodes.count(node)) {
        visited_nodes.insert(node);
        dot_emit_node_label(out, node);
    }
    if (visited_edges.count(node)) return;
    visited_edges.insert(node);

    string id = "n" + to_string((uintptr_t)node);

    // forward edges: inputs → node (white arrows)
    for (auto& inp : node->inputs) {
        if (!inp) continue;
        string child_id = "n" + to_string((uintptr_t)inp.get());
        out << "    " << child_id << " -> " << id
            << " [color=\"#FFFFFF\" penwidth=3 arrowsize=1.2];\n";
        dot_emit_node_full(out, inp.get(), visited_nodes, visited_edges);
    }

    // backward edges: node → outputs (cyan arrows)
    for (auto& outp : node->outputs) {
        if (outp.expired()) continue;
        string child_id = "n" + to_string((uintptr_t)outp.lock().get());
        out << "    " << id << " -> " << child_id
            << " [color=\"#00ffff\" penwidth=3 arrowsize=1.2];\n";
        dot_emit_node_full(out, outp.lock().get(), visited_nodes, visited_edges);
    }
}

static void find_roots(Node* node, unordered_set<Node*>& visited,
                       vector<Node*>& roots) {
    if (visited.count(node)) return;
    visited.insert(node);

    if (node->prim && node->prim->name() == "Leaf") {
        roots.push_back(node);
        return;
    }

    for (auto& inp : node->inputs)
        find_roots(inp.get(), visited, roots);
}

// void export_dot_full(Node* start, const char* filename) {
//     unordered_set<Node*> visited_roots;
//     vector<Node*> roots;
//     find_roots(start, visited_roots, roots);

//     ofstream out(filename);
//     out << "digraph FullGraph {\n"
//         << "    rankdir=BT;\n"
//         << "    bgcolor=\"#000000\";\n"
//         << "    pad=1.2;\n"
//         << "    nodesep=1.5;\n"
//         << "    ranksep=1.5;\n"
//         << "    node [fontname=\"Ubuntu\" fontsize=12 margin=\"0.2,0.1\" fontcolor=\"#FFFFFF\" width=0 height=0];\n"
//         << "    edge [fontname=\"Ubuntu\" fontsize=9];\n\n";

//     string start_id = "n" + to_string((uintptr_t)start);
//     out << "    " << start_id << " [penwidth=4 color=\"#FF5722\"];\n\n";

//     unordered_set<Node*> visited_nodes;
//     unordered_set<Node*> visited_edges;
//     for (auto* root : roots)
//         dot_emit_node_full(out, root, visited_nodes, visited_edges);

//     out << "}\n";
//     out.close();

//     // cout << "Full graph written to " << filename << endl;
//     // cout << "Render with:  dot -Tpdf " << filename << " -o full.pdf" << endl;
// }

void export_dot_full(Node* root, const char* filename) {

    ofstream out(filename);
    out << "digraph FullGraph {\n"
        << "    rankdir=BT;\n"
        << "    bgcolor=\"#000000\";\n"
        << "    pad=1.2;\n"
        << "    nodesep=1.5;\n"
        << "    ranksep=1.5;\n"
        << "    node [fontname=\"Ubuntu\" fontsize=12 margin=\"0.2,0.1\" fontcolor=\"#FFFFFF\" width=0 height=0];\n"
        << "    edge [fontname=\"Ubuntu\" fontsize=9];\n\n";

    string start_id = "n" + to_string((uintptr_t)root);
    out << "    " << start_id << " [penwidth=4 color=\"#FF5722\"];\n\n";

    unordered_set<Node*> visited_nodes;
    unordered_set<Node*> visited_edges;
    dot_emit_node_full(out, root, visited_nodes, visited_edges);

    out << "}\n";
    out.close();

    // cout << "Full graph written to " << filename << endl;
    // cout << "Render with:  dot -Tpdf " << filename << " -o full.pdf" << endl;
}
