#include <istream>
#include <sstream>
#include <algorithm>
#include "mymat.h"


using std::cout;
using std::endl;
using std::istream;
using std::ostream;
using std::stringstream;

ostream& operator<<(ostream &os, const Mat&m){
    for (size_t i = 0; i < m.row; i++){
        for (size_t j = 0; j < m.col; j++){
            os << m.data[i][j] << " ";
        }
        os << std::endl;
    }
    os << std::endl;
    return os;
}

istream& operator>>(istream &is, Mat&m){
    for (size_t i = 0; i < m.row; i++){
        for (size_t j = 0; j < m.col; j++){
            is >> m.data[i][j];
        }
    }
    return is;
}

// +
const Mat operator+(const Mat& m1, const Mat& m2){
    Mat t = m1;
    t += m2;
    return t;
}


// -
const Mat operator-(const Mat& m1, const Mat& m2){
    Mat t = m1;
    t -= m2;
    return t;
}

//constructor
Mat::Mat(){
    cout << "default constructor" << endl;
    row = 0;
    col = 0;
    data.clear();
}


Mat::Mat(size_t i, size_t j){
    row = i; col = j;
    std::vector<std::vector<int>> vdata(row, std::vector<int>(col, 0));
    data = std::move(vdata);
}

//copy constructor
Mat::Mat(const Mat& m){
    cout << "copy constructor" << endl;
    row = m.row; col = m.col;
    data = m.data;
}

//copy assignment
Mat& Mat::operator=(const Mat& m){
    cout << "copy assignment" << endl;
    row = m.row; col = m.col;
    data = m.data;
    return *this;
}

//destructor
Mat::~Mat(){
    data.clear();
}

//access element value
int& Mat::operator()(size_t i, size_t j){
    assert(i >= 0 && j >= 0 && i < row && j < col);
    return data[i][j];
}

const int& Mat::operator()(size_t i, size_t j) const{
    assert(i >= 0 && j >= 0 && i < row && j < col);
    return data[i][j];
}

//resize
void Mat::resize(size_t nr, size_t nc){
    data.resize(nr);
    for (size_t i = 0; i < nr; i++){
        data[i].resize(nc);
    }
    col = nc; row = nr;
}



// +=
Mat& Mat::operator+=(const Mat& m){
    if (row == m.row && col == m.col){
        for (size_t i = 0; i < row; i++)
        {
            for (size_t j = 0; j < col; j++)
                data[i][j] += m.data[i][j];
        }
    }
    else{
        std::cerr << "mat must be the same size." << std::endl;
    }
    return *this;
}

// -=
Mat& Mat::operator-=(const Mat& m){
    if (row == m.row && col == m.col){
        for (size_t i = 0; i < row; i++)
        {
            for (size_t j = 0; j < col; j++)
                data[i][j] -= m.data[i][j];
        }
    }
    else{
        std::cerr << "mat must be the same size." << std::endl;
    }
    return *this;
}

#if 1

int main(){

    Mat mat1(3, 4);
    Mat mat2(3, 4);

    for (size_t i = 0; i < mat1.rows(); i++){
        for (size_t j = 0; j < mat1.cols(); j++){
            mat1(i, j) = 1;
            mat2(i, j) = 3;
        }
    }
    std::cout << "mat1: " << std::endl << mat1;
    std::cout << "mat2: " << std::endl << mat2;

    Mat mat3 = (mat2 + mat1);
    std::cout << "mat3 = mat2 + mat1: " << std::endl << mat3;

    Mat mat4 = (mat3 + mat2 - mat1);
    std::cout << "mat4 = mat3 + mat2 - mat1: " << std::endl << mat4;

    stringstream ss;
    ss << mat1;
    ss >> mat4;
    std::cout << "mat4:" << std::endl << mat4;

    const Mat   mat6(mat4);
    std::cout << "const mat6:" << std::endl << mat6;
    cout << mat6(0, 0) << " " << mat6.rows() << " "<<mat6.cols()<<" ";

    Mat mat7 = mat2;
    std::cout << "mat7: " << std::endl << mat7;

    mat2(0, 0) = 11;
    std::cout << "mat7: " << std::endl << mat7;

    mat7.resize(2, 3);
    std::cout << "mat7.resize(2, 3): " << std::endl << mat7;

    mat7.resize(5, 6);
    std::cout << "mat7.resize(5, 6): " << std::endl << mat7;

    return 1;
}

#endif

