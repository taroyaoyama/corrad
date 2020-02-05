
#' Extended correlation anomaly detection (ECAD-I & ECAD-II)
#'
#' @param DA reference data matrix.
#' @param DB test data matrix.
#' @param rho penalty weight for graphical lasso.
#' @param dlt time interval of recorded data.
#' @param eliminate_phase boolean. defalut is FALSE.
#'   if TRUE, it executes foulier-amplitude-based ECAD (ECAD-II).
#' @param smooth boolean. if TRUE, apply parzen-window smoothing to DA & DB.
#' @param bandwidth if smooth == TRUE, it specifies bandwidth of parzen-window.
#' @importFrom glasso glasso
#' @export

ex_corrad <- function (DA, DB, rho, dlt, eliminate_phase = FALSE,
                       smooth = FALSE, bandwidth = 0.2) {

    DA <- scale(DA)
    DB <- scale(DB)

    M  <- ncol(DA)
    N  <- nrow(DA)
    N2 <- as.integer(N/2)

    freq <- 1:N2/N/dlt

    KAs <- array(0, c(M, M, N2))
    KBs <- array(0, c(M, M, N2))

    # ECAD-I
    if (!eliminate_phase) {

        for (i in 1:M) {

            for (j in 1:M) {

                KAs[i,j,] <- Re(psd_parzen(DA[,i], DA[,j], dt = dlt, smooth = FALSE)$power)/2/pi/2
                KBs[i,j,] <- Re(psd_parzen(DB[,i], DB[,j], dt = dlt, smooth = FALSE)$power)/2/pi/2

                if (smooth == TRUE) {

                    KAs[i,j,] <- parzen_window_smoothing(freq, KAs[i,j,], bandwidth = bandwidth)$power
                    KBs[i,j,] <- parzen_window_smoothing(freq, KBs[i,j,], bandwidth = bandwidth)$power
                }
            }
        }

    # ECAD-II
    } else {

        for (i in 1:M) {

            for (j in 1:M) {

                KAs[i,j,] <- Mod(psd_parzen(DA[,i], DA[,j], dt = dlt, smooth = FALSE)$power)/2/pi/2
                KBs[i,j,] <- Mod(psd_parzen(DB[,i], DB[,j], dt = dlt, smooth = FALSE)$power)/2/pi/2

                if (smooth == TRUE) {

                    KAs[i,j,] <- parzen_window_smoothing(freq, KAs[i,j,], bandwidth = bandwidth)$power
                    KBs[i,j,] <- parzen_window_smoothing(freq, KBs[i,j,], bandwidth = bandwidth)$power
                }
            }
        }
    }

    # anomaly scoring
    ASP <- array(0, c(N2, M))

    SgAs <- array(0, c(M, M, N2))
    LmAs <- array(0, c(M, M, N2))
    SgBs <- array(0, c(M, M, N2))
    LmBs <- array(0, c(M, M, N2))

    for (i in 1:N2) {

        SA <- KAs[,,i]
        SB <- KBs[,,i]

        glA <- glasso(SA, rho = rho)
        glB <- glasso(SB, rho = rho)

        ASP[i, ] <- corr_as(SgA = glA$w, LmA = glA$wi, SgB = glB$w, LmB = glB$wi)

        SgAs[,,i] <- glA$w
        LmAs[,,i] <- glA$wi
        SgBs[,,i] <- glB$w
        LmBs[,,i] <- glB$wi
    }

    AS <- apply(ASP, MARGIN = 2, sum)*(freq[2]-freq[1])*2*pi

    return(list(freq = freq, AScore = AS, ASpec = ASP,
                KA = KAs, KB = KBs, SgA = SgAs, LmA = LmAs, SgB = SgBs, LmB = LmBs))
}
