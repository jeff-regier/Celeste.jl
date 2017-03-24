require(argparse)
require(dplyr)
require(ggplot2)

parser <- ArgumentParser()
parser$add_argument('errors_csv')
arguments <- parser$parse_args()

data <- (
    read.csv(arguments$errors_csv)
    %>% mutate(z_score=error / posterior_std_err)
)

(ggplot(data, aes(sample=z_score))
    + stat_qq(shape='O')
    + geom_abline(slope=1, intercept=0, linetype='dashed')
    + facet_wrap(~ name, scales='free_y')
    + theme_bw()
    + ggtitle('Q-Q plot of errors for all sources')
)

trimmed_data <- (
    data
    %>% group_by(name)
    %>% filter(abs(z_score) < quantile(abs(z_score), 0.9))
)

(ggplot(trimmed_data, aes(sample=z_score))
    + stat_qq(shape='O')
    + geom_abline(slope=1, intercept=0, linetype='dashed')
    + facet_wrap(~ name, scales='free_y')
    + theme_bw()
    + ggtitle('Q-Q plot of middle 90% of errors')
)
