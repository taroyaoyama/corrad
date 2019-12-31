
#' Correlation anomaly detection
#'
#' @param DA reference data matrix.
#' @param DB test data matrix.
#' @param rho penalty weight for graphical lasso.
#' @param lag boolean. defalut is FALSE.
#'   if TRUE, it corrects time lag between time series
#'   by maximizing cross-correlation.
#' @param eliminate_phase boolean. defalut is FALSE.
#'   if TRUE, it executes foulier-amplitude-based CAD (CAD-II).
#' @importFrom glasso glasso
#' @export

corrad <- function (DA, DB, rho, lag = FALSE, eliminate_phase = FALSE) {

    # CAD-I
    if (eliminate_phase == FALSE) {

        if (lag == TRUE) {

            SA <- array(0, dim = c(ncol(DA), ncol(DA)))
            SB <- array(0, dim = c(ncol(DA), ncol(DA)))

            for (i in 1:ncol(DA)) {

                for (j in 1:ncol(DA)) {

                    SA[i, j] <- max(ccf(DA[,i], DA[,j], lag.max = 2500, plot = FALSE)$acf)
                    SB[i, j] <- max(ccf(DB[,i], DB[,j], lag.max = 2500, plot = FALSE)$acf)
                }
            }

        } else {

            SA <- cor(DA)
            SB <- cor(DB)
        }
    }

    # CAD-II (Fourier-amplitude based)
    if (eliminate_phase == FALSE) {

        if (lag == TRUE) {

            SA <- array(0, dim = c(ncol(DA), ncol(DA)))
            SB <- array(0, dim = c(ncol(DA), ncol(DA)))

            for (i in 1:ncol(DA)) {

                for (j in 1:ncol(DA)) {

                    SA[i, j] <- max(ccf(DA[,i], DA[,j], lag.max = 2500, plot = FALSE)$acf)
                    SB[i, j] <- max(ccf(DB[,i], DB[,j], lag.max = 2500, plot = FALSE)$acf)
                }
            }

        } else {

            DA <- scale(DA)
            DB <- scale(DB)

            FA <- Mod(mvfft(DA)/nrow(DA))
            FB <- Mod(mvfft(DB)/nrow(DB))

            SA <- t(FA) %*% FA
            SB <- t(FB) %*% FB
        }
    }

    # sparse structure learning using graphical lasso
    glA <- glasso(SA, rho)
    glB <- glasso(SB, rho)

    # anomaly scoring
    AS <- corr_as(SgA = glA$w, LmA = glA$wi, SgB = glB$w, LmB = glB$wi)

    return(list(AScore = AS, SA = SA, SB = SB,
                SgA = glA$w, LmA = glA$wi, SgB = glB$w, LmB = glB$wi))
}
