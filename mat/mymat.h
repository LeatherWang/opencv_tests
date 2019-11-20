#ifndef _MAT_H_
#define _MAT_H_

#include <iostream>
#include <ostream>
#include <vector>
#include <cstring>
#include <cassert>

//implement Mat class in c++

class Mat{
    friend  std::ostream& operator<<(std::ostream &os, const Mat &m);
    friend  std::istream& operator>>(std::istream &is, Mat &m);

public:
    typedef int value_type;
    typedef std::vector<int>::size_type size_type;

    //construct
    Mat();
    Mat(size_t i, size_t j);

    //copy constructor
    Mat(const Mat& m);

    //copy assignment
    Mat& operator=(const Mat& m);

    // +=
    Mat& operator+=(const Mat& m);

    // -=
    Mat& operator-=(const Mat& m);

    //destructor
    ~Mat();

    //access element value
    int& operator()(size_t i, size_t j);
    const int& operator()(size_t i, size_t j) const;


    //get row and col number
    const size_t rows() const{ return row; }
    const size_t cols() const{ return col; }

    //resize
    void resize(size_t nr, size_t nc);

private:

    size_t row;
    size_t col;
    std::vector<std::vector<int>> data;
};

#endif

