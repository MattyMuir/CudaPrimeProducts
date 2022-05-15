#pragma once
#include <string>
#include <iostream>
#include <fstream>

int IntInput(std::string message)
{
    std::string str;
    std::cout << message;
    getline(std::cin, str);
    return std::stoi(str, nullptr, 10);
}

void SaveRemainders(uint64_t start, uint64_t end, const uint64_t* saveRems, int num)
{
    std::ofstream out("C:\\Users\\matty\\source\\repos\\Prime-Products\\rems.log");
    out << start << '\n';
    out << end << '\n';
    for (int i = 0; i < num; i++)
        out << saveRems[i] << '\n';
}