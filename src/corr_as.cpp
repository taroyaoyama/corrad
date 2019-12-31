
#include <RcppEigen.h>

//' Calculating correlation anomaly score using KL divergence
//'
//' @param SgA,SgB covariance matrix of reference or test data.
//' @param LmA,LmB inverse covariance matrix (i.e. precision matrix)
//'   of reference or test data.
//' @export

//[[Rcpp::export]]
Eigen::VectorXd corr_as(
        Eigen::Map<Eigen::MatrixXd> SgA,
        Eigen::Map<Eigen::MatrixXd> SgB,
        Eigen::Map<Eigen::MatrixXd> LmA,
        Eigen::Map<Eigen::MatrixXd> LmB) {

    int M = SgA.rows();
    Eigen::VectorXd df = Eigen::VectorXd::Zero(M);

    double lmA, lmB, sgA, sgB, dAB, dBA;
    Eigen::MatrixXd LA(M - 1, M - 1);
    Eigen::MatrixXd LB(M - 1, M - 1);
    Eigen::MatrixXd lA(M - 1,     1);
    Eigen::MatrixXd lB(M - 1,     1);
    Eigen::MatrixXd WA(M - 1, M - 1);
    Eigen::MatrixXd WB(M - 1, M - 1);
    Eigen::MatrixXd wA(M - 1,     1);
    Eigen::MatrixXd wB(M - 1,     1);

    for (int i = 0; i < M; i++) {

        lmA = LmA(i, i);
        lmB = LmB(i, i);
        sgA = SgA(i, i);
        sgB = SgB(i, i);

        LA.block(0, 0,     i,     i) = LmA.block(  0,   0,     i,     i);
        LA.block(0, i,     i, M-i-1) = LmA.block(  0, i+1,     i, M-i-1);
        LA.block(i, 0, M-i-1,     i) = LmA.block(i+1,   0, M-i-1,     i);
        LA.block(i, i, M-i-1, M-i-1) = LmA.block(i+1, i+1, M-i-1, M-i-1);
        LB.block(0, 0,     i,     i) = LmB.block(  0,   0,     i,     i);
        LB.block(0, i,     i, M-i-1) = LmB.block(  0, i+1,     i, M-i-1);
        LB.block(i, 0, M-i-1,     i) = LmB.block(i+1,   0, M-i-1,     i);
        LB.block(i, i, M-i-1, M-i-1) = LmB.block(i+1, i+1, M-i-1, M-i-1);

        WA.block(0, 0,     i,     i) = SgA.block(  0,   0,     i,     i);
        WA.block(0, i,     i, M-i-1) = SgA.block(  0, i+1,     i, M-i-1);
        WA.block(i, 0, M-i-1,     i) = SgA.block(i+1,   0, M-i-1,     i);
        WA.block(i, i, M-i-1, M-i-1) = SgA.block(i+1, i+1, M-i-1, M-i-1);
        WB.block(0, 0,     i,     i) = SgB.block(  0,   0,     i,     i);
        WB.block(0, i,     i, M-i-1) = SgB.block(  0, i+1,     i, M-i-1);
        WB.block(i, 0, M-i-1,     i) = SgB.block(i+1,   0, M-i-1,     i);
        WB.block(i, i, M-i-1, M-i-1) = SgB.block(i+1, i+1, M-i-1, M-i-1);

        lA.block(0, 0,     i,     1) = LmA.block(  0,   i,     i,     1);
        lA.block(i, 0, M-i-1,     1) = LmA.block(i+1,   i, M-i-1,     1);
        lB.block(0, 0,     i,     1) = LmB.block(  0,   i,     i,     1);
        lB.block(i, 0, M-i-1,     1) = LmB.block(i+1,   i, M-i-1,     1);

        wA.block(0, 0,     i,     1) = SgA.block(  0,   i,     i,     1);
        wA.block(i, 0, M-i-1,     1) = SgA.block(i+1,   i, M-i-1,     1);
        wB.block(0, 0,     i,     1) = SgB.block(  0,   i,     i,     1);
        wB.block(i, 0, M-i-1,     1) = SgB.block(i+1,   i, M-i-1,     1);

        Eigen::MatrixXd tmp1(1, 1);
        Eigen::MatrixXd tmp2(1, 1);
        double tmp3;

        tmp1 = wA.transpose() * (lB - lA);
        tmp2 = 0.5 * ( (lB.transpose() * WA * lB).array() / lmB -
            (lA.transpose() * WA * lA).array() / lmA );
        tmp3 = 0.5 * ( log(lmA / lmB) + sgA * (lmB - lmA) );
        dAB  = (tmp1.array() + tmp2.array() + tmp3)(0, 0);

        tmp1 = wB.transpose() * (lA - lB);
        tmp2 = 0.5 * ( (lA.transpose() * WB * lA).array() / lmA -
            (lB.transpose() * WB * lB).array() / lmB );
        tmp3 = 0.5 * ( log(lmB / lmA) + sgB * (lmA - lmB) );
        dBA  = (tmp1.array() + tmp2.array() + tmp3)(0, 0);

        if (dAB > dBA)  df(i) = dAB;
        else            df(i) = dBA;

    }

    return df;
}
