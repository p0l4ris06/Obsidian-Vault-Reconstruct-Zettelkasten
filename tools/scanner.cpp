#include <iostream>
#include <string>
#include <vector>
#include <filesystem>
#include <fstream>
#include <regex>
#include <map>
#include <set>
#include <algorithm>
#include <cctype>

namespace fs = std::filesystem;

std::map<std::string, std::string> tag_map = {
    {"behavior", "behaviour"}, {"color", "colour"}, {"analyze", "analyse"}, {"center", "centre"}
};

struct HealthReport {
    int total_notes = 0;
    int fixed_links = 0;
    int fixed_tags = 0;
};

std::string normalize(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c){ return std::tolower(c); });
    s.erase(std::remove_if(s.begin(), s.end(), [](unsigned char c){ return std::isspace(c); }), s.end());
    return s;
}

void process_file(const fs::path& p, const std::set<std::string>& titles, const std::map<std::string, std::string>& title_norm_map, bool fix, HealthReport& report) {
    std::ifstream file(p, std::ios::in | std::ios::binary);
    if (!file.is_open()) return;

    std::string content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    file.close();

    bool modified = false;

    // 1. Fix Tags
    for (auto const& [from, to] : tag_map) {
        std::regex r("#" + from + "\\b", std::regex_constants::icase);
        std::string new_content = std::regex_replace(content, r, "#" + to);
        if (new_content != content) {
            content = new_content;
            report.fixed_tags++;
            modified = true;
        }
    }

    // 2. Fix Links
    std::regex link_regex(R"(\[\[([^\]|#]+))");
    auto search_content = content;
    auto links_begin = std::sregex_iterator(search_content.begin(), search_content.end(), link_regex);
    auto links_end = std::sregex_iterator();

    std::string new_content = content;
    for (std::sregex_iterator i = links_begin; i != links_end; ++i) {
        std::string target = (*i)[1].str();
        std::string target_norm = normalize(target);
        
        if (titles.find(target) == titles.end()) {
            if (title_norm_map.count(target_norm)) {
                std::string correct_title = title_norm_map.at(target_norm);
                // Regex replace to handle all instances
                std::regex r(R"(\[\[)" + target + R"(\]\])");
                new_content = std::regex_replace(new_content, r, "[[" + correct_title + "]]");
                report.fixed_links++;
                modified = true;
            }
        }
    }
    content = new_content;

    if (modified && fix) {
        std::ofstream out(p, std::ios::out | std::ios::binary);
        out << content;
        out.close();
    }
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: scanner <vault_path> [--fix]" << std::endl;
        return 1;
    }
    fs::path vault_path = argv[1];
    bool fix = (argc > 2 && std::string(argv[2]) == "--fix");

    if (!fs::exists(vault_path)) {
        std::cerr << "Error: Path does not exist: " << vault_path << std::endl;
        return 1;
    }

    HealthReport report;
    std::set<std::string> titles;
    std::map<std::string, std::string> title_norm_map;

    // First pass
    for (const auto& entry : fs::recursive_directory_iterator(vault_path)) {
        if (entry.is_regular_file()) {
            auto ext = entry.path().extension().string();
            // Case-insensitive check
            std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
            if (ext == ".md") {
                std::string title = entry.path().stem().string();
                titles.insert(title);
                title_norm_map[normalize(title)] = title;
                report.total_notes++;
            }
        }
    }

    // Second pass
    if (report.total_notes > 0) {
        for (const auto& entry : fs::recursive_directory_iterator(vault_path)) {
            if (entry.is_regular_file()) {
                auto ext = entry.path().extension().string();
                std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
                if (ext == ".md") {
                    process_file(entry.path(), titles, title_norm_map, fix, report);
                }
            }
        }
    }

    std::cout << "{\"total_notes\":" << report.total_notes 
              << ",\"fixed_links\":" << report.fixed_links 
              << ",\"fixed_tags\":" << report.fixed_tags << "}" << std::endl;

    return 0;
}
