#version 3: speeding up totally
#capping low values while remaining sum ==1
#not done: capping high_values (low_values are more important for robustness of system and with relative high capping of low values implicit capping of high_values is done as well)
#part above cap_low shrinks as much as is needed
rm(list=ls())

# cap_low <- 2e-6 
# cap_low <- 3e-6 

eps     <- .5e-15 #to check if everything is done with tolerance eps

base_dir = 'D:/Orca XL/Problems/PR0058 - Kaggle Data Science Bowl/' #on zeilvis

submission_dir <- paste0(base_dir, 'submissions/')
test <- F
if(test){
  ori <- 'systemtest'
}else{
  ori <- 'merge_5_model'
}
filename <- paste0(submission_dir, ori, '.csv')
submission_filename <- paste0(submission_dir, ori, '_cap_low3_', cap_low, '.csv')

submission   <- read.csv(filename, header=T, quote="")
#repair shrimp.like
rep_rownames <- names(submission)
rep_rownames <- gsub('shrimp.like', 'shrimp-like',  rep_rownames)
names(submission) <- rep_rownames

submission <- subset(submission, select=c('image', sort(names(submission[2:122]))))

starttime <- proc.time()
image <- submission[ , 1]                              #image vector
submission <- submission[ , 2:122]                     #keep rest
submission <- submission/rowSums(submission)           #if rowsum wasn't 1 make it 1
sub_min_cap <- submission - cap_low                    #subtract cap
sub_for_sum_above_cap <- sub_min_cap 
sub_for_shortage      <- sub_min_cap
sub_for_sum_above_cap[sub_for_sum_above_cap < 0] <- 0  #set min to 0
sums_above_cap <- rowSums(sub_for_sum_above_cap)
sub_for_shortage[sub_for_shortage > 0] <- 0            #set max to 0
shortages <- rowSums(sub_for_shortage)
shrinkages <- (sums_above_cap + shortages)/sums_above_cap #shortages are neg
new_sub <- (sub_for_sum_above_cap * shrinkages) + cap_low
submission <- cbind(image, new_sub)

#self-test
if(test){
  rows_in_sub <- 4973
}else{
  rows_in_sub <- 130400
}
rows_to_check <- sample.int(rows_in_sub, 100)
for(i in rows_to_check){
  row <- unlist(submission[i,2:122])
  if(abs(sum(row) - 1) >= eps){
    print(i)
    print(submission[i,])
  }
  stopifnot(abs(sum(row) - 1) < eps)
  stopifnot(min(row) >= cap_low)
}

write.csv(submission, submission_filename, row.names=F, quote=F)

intermediate_time <- proc.time()
time_used_so_far <- intermediate_time[3]-starttime[3]
print(paste('done in', time_used_so_far, 'seconds'))


