#!/usr/bin/env Rscript
# FLMM arm of the tonic-response benchmark (photosvi.experiments.tonic_benchmark).
#
#   Rscript scripts/cohort_tonic_flmm.R <in.csv> <out.json>
#
# <in.csv>: one row per session, columns animal, cond (0 vehicle, 1 treated), Y_1..Y_L (mean
# dF/F per 1-min bin aligned to the injection). Fits the functional linear mixed model
# Y(s) ~ cond + (1 | animal) with fastFMM::fui (fast univariate inference: Cui et al. 2022,
# J. Comput. Graph. Stat.; applied to fiber photometry by Loewinger et al. 2025, eLife),
# analytic variance as in the package's photometry tutorial, and writes the condition
# coefficient beta_cond(s), its pointwise standard error and the joint-band quantile.
suppressPackageStartupMessages({library(fastFMM); library(jsonlite)})
args <- commandArgs(trailingOnly = TRUE)
dat <- read.csv(args[1])
ycols <- grep("^Y_", names(dat))
Y <- as.matrix(dat[, ycols])
df <- data.frame(animal = factor(dat$animal), cond = dat$cond)
df$Y <- Y
fit <- fui(Y ~ cond + (1 | animal), data = df, analytic = TRUE, parallel = FALSE, silent = TRUE)
k <- which(rownames(fit$betaHat) == "cond")
# fastFMM 1.0.x names it betaHat_var (older versions betaHat.var).
V <- if (!is.null(fit$betaHat_var)) fit$betaHat_var else fit$betaHat.var
se <- sqrt(diag(V[, , k]))
out <- list(beta_cond = as.numeric(fit$betaHat[k, ]), se_cond = as.numeric(se),
            qn_cond = as.numeric(fit$qn[k]), argvals = as.numeric(fit$argvals))
write_json(out, args[2], digits = NA, auto_unbox = TRUE)
