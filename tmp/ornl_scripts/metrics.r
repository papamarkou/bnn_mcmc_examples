ess <- function(chain) {
  return(length(chain)*var(chain)/initseq(chain)$var.con)
}
