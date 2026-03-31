#include "graph_viz.h"
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>

using namespace std;

// ── ANSI colors ──
namespace clr {
    const char* reset   = "\033[0m";
    const char* bold    = "\033[1m";
    const char* dim     = "\033[2m";
    const char* cyan    = "\033[36m";
    const char* green   = "\033[32m";
    const char* yellow  = "\033[33m";
    const char* magenta = "\033[35m";
    const char* red     = "\033[31m";
    const char* blue    = "\033[34m";
    const char* bcyan   = "\033[1;36m";
    const char* bgreen  = "\033[1;32m";
    const char* byellow = "\033[1;33m";
    const char* bmagenta= "\033[1;35m";
    const char* bblue   = "\033[1;34m";
    const char* bred    = "\033[1;31m";
    const char* gray    = "\033[90m";
}

static const char* op_symbol(OpType op) {
    switch (op) {
        case OpType::Add:    return "+";
        case OpType::Mul:    return "✕";
        case OpType::MatMul: return "⊗";
        case OpType::Sum:    return "Σ";
        case OpType::Max:    return "↑";
        case OpType::Min:    return "↓";
        case OpType::ReLU:   return "⌐";
    }
    return "?";
}

static const char* op_name(OpType op) {
    switch (op) {
        case OpType::Add:    return "Add";
        case OpType::Mul:    return "Mul";
        case OpType::MatMul: return "MatMul";
        case OpType::Sum:    return "Sum";
        case OpType::Max:    return "Max";
        case OpType::Min:    return "Min";
        case OpType::ReLU:   return "ReLU";
    }
    return "??";
}

static const char* op_color(OpType op) {
    switch (op) {
        case OpType::Add:    return clr::bcyan;
        case OpType::Mul:    return clr::bmagenta;
        case OpType::MatMul: return clr::bblue;
        case OpType::Sum:    return clr::byellow;
        case OpType::Max:    return clr::bred;
        case OpType::Min:    return clr::bgreen;
        case OpType::ReLU:   return clr::byellow;
    }
    return clr::reset;
}

static string shape_str(const ShapePtr& meta) {
    ostringstream ss;
    ss << "[";
    for (int i = 0; i < meta->ndim; i++) {
        if (i) ss << ", ";
        ss << meta->shape()[i];
    }
    ss << "]";
    return ss.str();
}

static string strides_str(const ShapePtr& meta) {
    ostringstream ss;
    ss << "[";
    for (int i = 0; i < meta->ndim; i++) {
        if (i) ss << ", ";
        ss << meta->strides()[i];
    }
    ss << "]";
    return ss.str();
}

// count visible characters (skip ANSI escapes)
static int visible_len(const string& s) {
    int len = 0;
    bool in_esc = false;
    for (char c : s) {
        if (c == '\033') in_esc = true;
        else if (in_esc && c == 'm') in_esc = false;
        else if (!in_esc) len++;
    }
    return len;
}

struct BoxLines {
    vector<string> lines;
    int width; // visible width (inner content)
};

static BoxLines make_box(Node* node) {
    vector<pair<string, string>> rows; // label, value (both may have ANSI)
    string title;
    const char* border_color;

    if (node->kind == NodeKind::Internal) {
        InternalNode* inode = static_cast<InternalNode*>(node);
        string sym = op_symbol(inode->op);
        string name = op_name(inode->op);
        border_color = op_color(inode->op);
        title = string(border_color) + clr::bold + " " + sym + " " + name + " " + clr::reset;
        rows.push_back({string(clr::dim) + "Shape:" + clr::reset,
                        string(clr::yellow) + shape_str(node->h_meta) + clr::reset});
        rows.push_back({string(clr::dim) + "Strides:" + clr::reset,
                        string(clr::dim) + strides_str(node->h_meta) + clr::reset});
        rows.push_back({string(clr::dim) + "Uses:" + clr::reset,
                        string(clr::cyan) + to_string(node->use_count) + clr::reset});
    } else {
        border_color = clr::green;
        SourceNode* snode = static_cast<SourceNode*>(node);
        title = string(clr::bgreen) + " ▣ Source " + clr::reset;
        rows.push_back({string(clr::dim) + "Shape:" + clr::reset,
                        string(clr::yellow) + shape_str(node->h_meta) + clr::reset});
        rows.push_back({string(clr::dim) + "Strides:" + clr::reset,
                        string(clr::dim) + strides_str(node->h_meta) + clr::reset});
        rows.push_back({string(clr::dim) + "Uses:" + clr::reset,
                        string(clr::cyan) + to_string(snode->use_count) + clr::reset});
        if (snode->h_requiers_grad)
            rows.push_back({string(clr::bgreen) + "requires_grad" + clr::reset, ""});
    }

    // compute max visible width
    int title_vis = visible_len(title);
    int max_w = title_vis;
    for (auto& [l, v] : rows) {
        int row_vis = visible_len(l) + 2 + visible_len(v); // "label  value"
        if (row_vis > max_w) max_w = row_vis;
    }
    max_w += 2; // padding

    BoxLines box;
    box.width = max_w;

    // top border
    {
        string top = string(border_color) + "╭";
        string title_padded = title;
        int fill = max_w - title_vis;
        for (int i = 0; i < fill; i++) top += "─";
        top = string(border_color) + "╭─" + clr::reset + title_padded + string(border_color);
        int remaining = max_w - title_vis - 1;
        for (int i = 0; i < remaining; i++) top += "─";
        top += "╮";
        top += clr::reset;
        box.lines.push_back(top);
    }

    // content rows
    for (auto& [l, v] : rows) {
        string row = string(border_color) + "│" + clr::reset + " ";
        row += l;
        int label_vis = visible_len(l);
        // pad between label and value
        int gap = 10 - label_vis; // align values
        for (int i = 0; i < gap; i++) row += " ";
        row += v;
        int used = 1 + label_vis + gap + visible_len(v);
        int pad = max_w + 1 - used;
        for (int i = 0; i < pad; i++) row += " ";
        row += string(border_color) + "│" + clr::reset;
        box.lines.push_back(row);
    }

    // bottom border
    {
        string bot = string(border_color) + "╰";
        for (int i = 0; i < max_w + 1; i++) bot += "─";
        bot += "╯";
        bot += clr::reset;
        box.lines.push_back(bot);
    }

    return box;
}

// recursive tree drawing
static void draw_node(Node* node, const string& prefix, bool is_last, bool is_root) {
    BoxLines box = make_box(node);

    // connector from parent
    string connector_prefix;
    if (is_root) {
        connector_prefix = prefix;
    } else {
        connector_prefix = prefix + (is_last ? "  " : "│ ");
    }

    // draw box
    for (int i = 0; i < (int)box.lines.size(); i++) {
        if (is_root) {
            cout << prefix << box.lines[i] << endl;
        } else if (i == 0) {
            cout << prefix << (is_last ? "└─ " : "├─ ") << box.lines[i] << endl;
        } else {
            cout << connector_prefix << "   " << box.lines[i] << endl;
        }
    }

    // draw children if internal
    if (node->kind == NodeKind::Internal) {
        InternalNode* inode = static_cast<InternalNode*>(node);
        string child_prefix = is_root ? prefix : connector_prefix;
        child_prefix += "   ";

        for (int i = 0; i < inode->num_inputs; i++) {
            bool last_child = (i == inode->num_inputs - 1);

            // draw connecting wire
            cout << child_prefix << (last_child ? "  " : "│ ") << endl;

            draw_node(inode->inputs[i].get(), child_prefix, last_child, false);
        }
    }
}

void visualize(const std::shared_ptr<Node>& root) {
    cout << endl;
    cout << clr::dim << "  ╶──── Computation Graph ────╴" << clr::reset << endl;
    cout << endl;
    draw_node(root.get(), "  ", true, true);
    cout << endl;
}

// ── Graphviz DOT export ──

static const char* dot_op_color(OpType op) {
    switch (op) {
        case OpType::Add:    return "#4FC3F7";
        case OpType::Mul:    return "#CE93D8";
        case OpType::MatMul: return "#90CAF9";
        case OpType::Sum:    return "#FFD54F";
        case OpType::Max:    return "#EF9A9A";
        case OpType::Min:    return "#A5D6A7";
        case OpType::ReLU:   return "#FFCC80";
    }
    return "#EEEEEE";
}

static void dot_emit_node(ofstream& out, Node* node, unordered_set<Node*>& visited) {
    if (visited.count(node)) return;
    visited.insert(node);

    string id = "n" + to_string((uintptr_t)node);
    string shape_s = shape_str(node->h_meta);

    if (node->kind == NodeKind::Internal) {
        InternalNode* inode = static_cast<InternalNode*>(node);
        string sym = op_symbol(inode->op);
        string name = op_name(inode->op);
        const char* color = dot_op_color(inode->op);

        string strides_s = strides_str(node->h_meta);
        out << "    " << id << " [\n"
            << "        label=<"
            << "<TABLE BORDER=\"0\" CELLBORDER=\"0\" CELLSPACING=\"4\">"
            << "<TR><TD COLSPAN=\"2\"><FONT POINT-SIZE=\"18\">" << sym << "</FONT> <B>" << name << "</B></TD></TR>"
            << "<TR><TD ALIGN=\"LEFT\"><FONT POINT-SIZE=\"10\" COLOR=\"#555555\">Shape: " << shape_s << "</FONT></TD></TR>"
            << "<TR><TD ALIGN=\"LEFT\"><FONT POINT-SIZE=\"10\" COLOR=\"#555555\">Strides: " << strides_s << "</FONT></TD></TR>"
            << "<TR><TD ALIGN=\"LEFT\"><FONT POINT-SIZE=\"10\" COLOR=\"#2196F3\">Uses: " << node->use_count << "</FONT></TD></TR>"
            << "</TABLE>>\n"
            << "        shape=box style=\"rounded,filled\" fillcolor=\"" << color << "\"\n"
            << "        penwidth=2 color=\"" << color << "30\"\n"
            << "    ];\n";

        for (int i = 0; i < inode->num_inputs; i++) {
            string child_id = "n" + to_string((uintptr_t)inode->inputs[i].get());
            out << "    " << child_id << " -> " << id
                << " [color=\"#666666\" penwidth=1.5 arrowsize=0.8];\n";
            dot_emit_node(out, inode->inputs[i].get(), visited);
        }
    } else {
        SourceNode* snode = static_cast<SourceNode*>(node);
        const char* color = snode->h_requiers_grad ? "#C8E6C9" : "#E0E0E0";
        const char* border = snode->h_requiers_grad ? "#66BB6A" : "#BDBDBD";
        string grad_label = snode->h_requiers_grad ? "grad: yes" : "";

        string strides_s = strides_str(node->h_meta);
        out << "    " << id << " [\n"
            << "        label=<"
            << "<TABLE BORDER=\"0\" CELLBORDER=\"0\" CELLSPACING=\"2\">"
            << "<TR><TD><B>Source</B></TD></TR>"
            << "<TR><TD ALIGN=\"LEFT\"><FONT POINT-SIZE=\"10\" COLOR=\"#555555\">Shape: " << shape_s << "</FONT></TD></TR>"
            << "<TR><TD ALIGN=\"LEFT\"><FONT POINT-SIZE=\"10\" COLOR=\"#555555\">Strides: " << strides_s << "</FONT></TD></TR>"
            << "<TR><TD ALIGN=\"LEFT\"><FONT POINT-SIZE=\"10\" COLOR=\"#2196F3\">Uses: " << snode->use_count << "</FONT></TD></TR>";
        if (snode->h_requiers_grad)
            out << "<TR><TD ALIGN=\"LEFT\"><FONT POINT-SIZE=\"9\" COLOR=\"#388E3C\">requires_grad</FONT></TD></TR>";
        out << "</TABLE>>\n"
            << "        shape=box style=\"rounded,filled\" fillcolor=\"" << color << "\"\n"
            << "        penwidth=2 color=\"" << border << "\"\n"
            << "    ];\n";
    }
}

void export_dot(const std::shared_ptr<Node>& root, const char* filename) {
    ofstream out(filename);
    out << "digraph ComputationGraph {\n"
        << "    rankdir=BT;\n"
        << "    bgcolor=\"#FAFAFA\";\n"
        << "    pad=0.5;\n"
        << "    nodesep=0.6;\n"
        << "    ranksep=0.8;\n"
        << "    node [fontname=\"Helvetica\" fontsize=12 margin=\"0.2,0.1\"];\n"
        << "    edge [fontname=\"Helvetica\" fontsize=9];\n\n";

    unordered_set<Node*> visited;
    dot_emit_node(out, root.get(), visited);

    out << "}\n";
    out.close();

    cout << "Graph written to " << filename << endl;
    cout << "Render with:  dot -Tpng " << filename << " -o graph.png" << endl;
}

// ── Backward traversal (topological sort) with visualization ──

static void topo_build(Node* node, unordered_set<Node*>& visited, vector<Node*>& order) {
    if (visited.count(node)) return;
    visited.insert(node);

    if (node->kind == NodeKind::Internal) {
        InternalNode* inode = static_cast<InternalNode*>(node);
        for (int i = 0; i < inode->num_inputs; i++)
            topo_build(inode->inputs[i].get(), visited, order);
    }
    order.push_back(node);
}

void backward_viz(const std::shared_ptr<Node>& root) {
    // topological sort
    unordered_set<Node*> visited;
    vector<Node*> order;
    topo_build(root.get(), visited, order);

    // backward = reverse topological order
    cout << endl;
    cout << clr::dim << "  ╶──── Backward Pass ────╴" << clr::reset << endl;
    cout << endl;

    for (int step = (int)order.size() - 1; step >= 0; step--) {
        Node* node = order[step];
        int idx = (int)order.size() - 1 - step;

        if (node->kind == NodeKind::Internal) {
            InternalNode* inode = static_cast<InternalNode*>(node);
            const char* color = op_color(inode->op);

            cout << "  " << clr::dim << "[" << idx << "]" << clr::reset
                 << "  " << color << clr::bold << op_symbol(inode->op) << " " << op_name(inode->op) << clr::reset << endl;
            cout << "       " << clr::dim << "Shape:   " << clr::reset
                 << clr::yellow << shape_str(node->h_meta) << clr::reset << endl;
            cout << "       " << clr::dim << "Strides: " << clr::reset
                 << clr::dim << strides_str(node->h_meta) << clr::reset << endl;
            cout << "       " << clr::dim << "Uses:    " << clr::reset
                 << clr::cyan << node->use_count << clr::reset << endl;

            // show which inputs this node pulls grad from
            cout << "       " << clr::dim << "inputs:  " << clr::reset;
            for (int i = 0; i < inode->num_inputs; i++) {
                if (i) cout << ", ";
                // find this input's position in topo order
                for (int j = 0; j < (int)order.size(); j++) {
                    if (order[j] == inode->inputs[i].get()) {
                        cout << clr::dim << "→ [" << ((int)order.size() - 1 - j) << "]" << clr::reset;
                        break;
                    }
                }
            }
            cout << endl;

        } else {
            SourceNode* snode = static_cast<SourceNode*>(node);
            cout << "  " << clr::dim << "[" << idx << "]" << clr::reset
                 << "  " << clr::bgreen << "▣ Source" << clr::reset << endl;
            cout << "       " << clr::dim << "Shape:   " << clr::reset
                 << clr::yellow << shape_str(node->h_meta) << clr::reset << endl;
            cout << "       " << clr::dim << "Strides: " << clr::reset
                 << clr::dim << strides_str(node->h_meta) << clr::reset << endl;
            cout << "       " << clr::dim << "Uses:    " << clr::reset
                 << clr::cyan << snode->use_count << clr::reset << endl;
            if (snode->h_requiers_grad)
                cout << "       " << clr::bgreen << "requires_grad" << clr::reset << endl;
        }

        if (step > 0)
            cout << "       " << clr::dim << "│" << clr::reset << endl;
    }
    cout << endl;
}
