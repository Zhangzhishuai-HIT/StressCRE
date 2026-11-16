# One DESeq2 test per stress-vs-control contrast (treated vs control).
# Time is a covariate only when every time level has both arms; otherwise ~ condition.
# Usage: Rscript de_run.R <work_dir> <n_cores>
suppressPackageStartupMessages({library(DESeq2); library(parallel)})
args <- commandArgs(trailingOnly = TRUE)
W <- args[1]; NC <- as.integer(args[2])
man <- read.delim(file.path(W, "CONTRASTS.tsv"), stringsAsFactors = FALSE)

one <- function(cid) {
  d <- file.path(W, cid)
  out <- file.path(d, "de.tsv.gz")
  if (file.exists(out)) return(NULL)
  cts <- as.matrix(read.delim(gzfile(file.path(d, "counts.tsv.gz")), row.names = 1, check.names = FALSE))
  col <- read.delim(file.path(d, "coldata.tsv"), row.names = 1, stringsAsFactors = FALSE)
  col <- col[colnames(cts), , drop = FALSE]
  col$condition <- factor(col$condition, levels = c("control", "treated"))
  col$time <- factor(ifelse(is.na(col$time), "NA", as.character(col$time)))
  tab <- table(col$time, col$condition)
  use_time <- nlevels(col$time) >= 2 && all(tab > 0)
  keep <- rowSums(cts >= 10) >= 2
  cts <- cts[keep, , drop = FALSE]
  form <- if (use_time) ~ time + condition else ~ condition
  dds <- DESeqDataSetFromMatrix(cts, col, form)
  dds <- tryCatch(DESeq(dds, quiet = TRUE),
                  error = function(e) DESeq(dds, quiet = TRUE, fitType = "mean"))
  res <- as.data.frame(results(dds, contrast = c("condition", "treated", "control")))
  res$gene <- rownames(res)
  con <- gzfile(out, "w")
  write.table(res[, c("gene", "baseMean", "log2FoldChange", "lfcSE", "pvalue", "padj")],
              con, sep = "\t", quote = FALSE, row.names = FALSE)
  close(con)
  sprintf("%s\t%d\t%d\t%d\t%s", cid, nrow(res),
          sum(!is.na(res$padj) & res$padj < 0.05 & res$log2FoldChange > 1),
          sum(!is.na(res$padj) & res$padj < 0.05 & res$log2FoldChange < -1),
          if (use_time) "time+condition" else "condition")
}
res <- mclapply(man$contrast, function(c) tryCatch(one(c), error = function(e) paste(c, "ERROR", conditionMessage(e))),
                mc.cores = NC, mc.preschedule = FALSE)
lines <- unlist(res)
cat(lines, sep = "\n")
write(c("contrast\ttested\tup\tdown\tdesign", lines[!grepl("ERROR", lines)]),
      file.path(W, "DE_SUMMARY.new.tsv"))
cat("done\n")
