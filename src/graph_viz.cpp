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

static const char* op_color(const string& name) {
    if (name == "add")    return "#4FC3F7";
    if (name == "sub")    return "#80DEEA";
    if (name == "mul")    return "#CE93D8";
    if (name == "matmul") return "#90CAF9";
    if (name == "neg")    return "#F48FB1";
    if (name == "exp")    return "#FFE082";
    if (name == "log")    return "#A5D6A7";
    if (name == "relu")   return "#FFCC80";
    if (name == "sum_ld") return "#FFD54F";
    if (name == "div")    return "#B0BEC5";
    if (name == "pow")    return "#BCAAA4";
    if (name == "skip")   return "#B3E5FC";
    return "#EEEEEE";
}

static bool is_leaf(Node* node) {
    return node->inputs.empty();
}

static void dot_emit_node_label(ofstream& out, Node* node) {
    string id = "n" + to_string((uintptr_t)node);
    string shape_s = shape_str(node);

    if (!is_leaf(node)) {
        string sym  = node->prim ? escape_html(node->prim->symbol()) : "?";
        string name = node->prim ? node->prim->name() : "unknown";
        const char* color = op_color(name);

        string label_row;
        if (!node->label.empty())
            label_row = "<TR><TD ALIGN=\"LEFT\">"
                        "<FONT FACE=\"iA Writer Mono S\" POINT-SIZE=\"14\" COLOR=\"#ffffff\">label = </FONT>"
                        "<FONT FACE=\"iA Writer Mono S\" POINT-SIZE=\"14\" COLOR=\"#ffff00\">"
                        + escape_html(node->label) + "</FONT></TD></TR>";

        out << "    " << id << " [\n"
            << "        label=<"
            << "<TABLE BORDER=\"0\" CELLBORDER=\"0\" CELLSPACING=\"4\">"
            << "<TR><TD COLSPAN=\"2\">"
            << "<FONT POINT-SIZE=\"20\" COLOR=\"#ffffff\">" << sym << " </FONT>"
            << "<FONT POINT-SIZE=\"18\" COLOR=\"#ffffff\"><B>" << name << "</B></FONT>"
            << "</TD></TR>"
            << label_row
            << "<TR><TD ALIGN=\"LEFT\">"
            << "<FONT FACE=\"iA Writer Mono S\" POINT-SIZE=\"14\" COLOR=\"#ffffff\">Shape: </FONT>"
            << "<FONT FACE=\"iA Writer Mono S\" POINT-SIZE=\"14\" COLOR=\"#04ff00\">" << shape_s << "</FONT>"
            << "</TD></TR>"
            << "<TR><TD ALIGN=\"LEFT\">"
            << "<FONT FACE=\"iA Writer Mono S\" POINT-SIZE=\"14\" COLOR=\"#ffffff\">transposed = </FONT>"
            << "<FONT FACE=\"iA Writer Mono S\" POINT-SIZE=\"14\" COLOR=\"#ff00c8\">"
            << (node->transposed ? "true" : "false") << "</FONT>"
            << "</TD></TR>"
            << "</TABLE>>\n"
            << "        shape=box style=\"rounded\" width=0 height=0 margin=\"0.1\"\n"
            << "        penwidth=2 color=\"#FFFFFF\"\n"
            << "    ];\n";
    } else {
        string label_row;
        if (!node->label.empty())
            label_row = "<TR><TD ALIGN=\"LEFT\">"
                        "<FONT FACE=\"iA Writer Mono S\" POINT-SIZE=\"14\" COLOR=\"#ffffff\">label = </FONT>"
                        "<FONT FACE=\"iA Writer Mono S\" POINT-SIZE=\"14\" COLOR=\"#ffff00\">"
                        + escape_html(node->label) + "</FONT></TD></TR>";
        
        bool scalar = shape_s == "[]";
        string shape_or_scalar_row;
        if (!scalar)
            shape_or_scalar_row = "<TR><TD ALIGN=\"LEFT\">"
                                "<FONT FACE=\"iA Writer Mono S\" POINT-SIZE=\"14\" COLOR=\"#ffffff\">Shape: </FONT>"
                                "<FONT FACE=\"iA Writer Mono S\" POINT-SIZE=\"14\" COLOR=\"#04ff00\">" + shape_s + "</FONT>"
                                "</TD></TR>";
        else
            shape_or_scalar_row = "<TR><TD ALIGN=\"LEFT\">"
                                "<FONT FACE=\"iA Writer Mono S\" POINT-SIZE=\"14\" COLOR=\"#ffffff\">Value: </FONT>"
                                "<FONT FACE=\"iA Writer Mono S\" POINT-SIZE=\"14\" COLOR=\"#04ff00\">" 
                                + to_string((int)static_cast<ScalarNode*>(node)->value) + "</FONT>"
                                "</TD></TR>";

            
        out << "    " << id << " [\n"
            << "        label=<"
            << "<TABLE BORDER=\"0\" CELLBORDER=\"0\" CELLSPACING=\"2\">"
            << "<TR><TD><FONT POINT-SIZE=\"18\" COLOR=\"#ffffff\"><B>" << (scalar ? "Scalar" : "Leaf") << "</B></FONT></TD></TR>"
            << label_row
            << shape_or_scalar_row;
        if (node->requires_grad)
            out << "<TR><TD ALIGN=\"LEFT\">"
                << "<FONT FACE=\"iA Writer Mono S\" POINT-SIZE=\"14\" COLOR=\"#ffffff\">requires_grad = </FONT>"
                << "<FONT FACE=\"iA Writer Mono S\" POINT-SIZE=\"14\" COLOR=\"#ff00c8\">true</FONT>"
                << "</TD></TR>";
        out << "</TABLE>>\n"
            << "        shape=box style=\"rounded\"\n"
            << "        penwidth=2 color=\"#FFFFFF\"\n"
            << "    ];\n";
    }
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
        if (!outp) continue;
        string child_id = "n" + to_string((uintptr_t)outp.get());
        out << "    " << id << " -> " << child_id
            << " [color=\"#00ffff\" penwidth=3 arrowsize=1.2];\n";
        dot_emit_node_full(out, outp.get(), visited_nodes, visited_edges);
    }
}

static void find_roots(Node* node, unordered_set<Node*>& visited,
                       vector<Node*>& roots) {
    if (visited.count(node)) return;
    visited.insert(node);

    if (is_leaf(node)) {
        roots.push_back(node);
        return;
    }

    for (auto& inp : node->inputs)
        find_roots(inp.get(), visited, roots);
}

void export_dot_full(Node* start, const char* filename) {
    unordered_set<Node*> visited_roots;
    vector<Node*> roots;
    find_roots(start, visited_roots, roots);

    ofstream out(filename);
    out << "digraph FullGraph {\n"
        << "    rankdir=BT;\n"
        << "    bgcolor=\"#000000\";\n"
        << "    pad=1.2;\n"
        << "    nodesep=1.5;\n"
        << "    ranksep=1.5;\n"
        << "    node [fontname=\"Ubuntu\" fontsize=12 margin=\"0.2,0.1\" fontcolor=\"#FFFFFF\" width=0 height=0];\n"
        << "    edge [fontname=\"Ubuntu\" fontsize=9];\n\n";

    string start_id = "n" + to_string((uintptr_t)start);
    out << "    " << start_id << " [penwidth=4 color=\"#FF5722\"];\n\n";

    unordered_set<Node*> visited_nodes;
    unordered_set<Node*> visited_edges;
    for (auto* root : roots)
        dot_emit_node_full(out, root, visited_nodes, visited_edges);

    out << "}\n";
    out.close();

    cout << "Full graph written to " << filename << endl;
    cout << "Render with:  dot -Tpdf " << filename << " -o full.pdf" << endl;
}
