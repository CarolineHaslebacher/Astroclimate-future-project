
##### plot densities from data
# define function

density_plots_insitu <- function(base_path, csites, cpressure) {
  
  par(mfrow=c(8,2)) # , mai=c(0.1,0.1,0.1,0.1)) 
  for (idx in seq_along(csites)) {
    #, mai=c(0.1,0.1,0.1,0.1)) # , mai=c(0.1,0.1,0.1,0.1)
    # layout(barmode = 'stack',autosize=F,yaxis=list(title=Title,automargin=TRUE),xaxis=list(automargin=T))
    
    for (forc in cforcing) {
      # path = paste(base_path, csites[idx], forc, cpressure[idx], "_5month_index_rolling_dataset.csv"  , sep='') # Siding_SpringSSTfuture_925_5month_index_rolling_dataset.csv
      path = paste(base_path, csites[idx],'_', forc,   cpressure[idx], "_monthly_timeseries.csv"  , sep='')
      
      # read in data
      
      d <- read.csv(path, header=TRUE, sep=",")
      
      pdf(paste(base_path, 'density_plots/','Density_plot_', csites[idx], '_', forc, cpressure[idx] ,"_monthly_timeseries.pdf" , sep='')) 
      
      # plot( temp ~ month_index , data=d , col=col.alpha(rangi2,0.8),  cex=1, pch=19 , ylab = 'Temperature [deg C]',font.main = 1, main=paste(csites[idx], forc, sep=' ')) # 
      par(mfrow=c(1,1))
      dens(d$temp)
      title(main = csites[idx])
      
      dev.off()
    }
  }
}






#################


############333

#########
# map fit (sine function)
# maybe I forgot some 
map_fit_sinusoidal <- function(base_path, csites, cpressure, variable) {
for (forc in cforcing){
  
  pdf(width=8, height=18, pointsize=10, file=paste(base_path, variable, '_', forc, "_Linear_fit_monthly_timeseries.pdf" , sep=''))
  par(mfrow=c(8,1)) # , mai=c(0.1,0.1,0.1,0.1)) 
  
  for (idx in seq_along(csites)) {
    # path = paste(base_path, csites[idx], forc, cpressure[idx], "_5month_index_rolling_dataset.csv"  , sep='') # Siding_SpringSSTfuture_925_5month_index_rolling_dataset.csv
    path = paste(base_path, csites[idx],'_', forc,   cpressure[idx], "_monthly_timeseries.csv"  , sep='')
    
    print(path)
    # path_list <- append(path_list, path) 
    
    # read in data
    d <- read.csv(path, header=TRUE, sep=",")
    
    # define time with as.numeric(rownames(d))
    d$month_index <- as.numeric(rownames(d))
    
    if (variable == 'total_cloud_cover'){
      # convert decimal to percentages of cloud cover
      d$temp <- d$temp * 100
    }
    
    # define m_prior and a_prior (now if we have the rowname, it makes sense to set m_prior to 0 and a_prior to the mean value)
    m_prior = 0
    a_prior_mean = mean(d$temp)-m_prior
    print(a_prior_mean)
    
    # map (maximum a posteriori estimate; uses quadratic approximation)
    # what this blackbox actually does: 
    # we tell the model that the prior distribution is composed of a normal distribution with mu and sigma, mu being the linear regression line
    # sigma is the standard deviation of the normal distribution of the prior
    # sigma is retrieved from a uniform distribution. e.g. if sigma ~ dunif(0 , 10), this means that 95% of the variables lie within 20 units around the mean value
    # mu is retrieved from a normal distribution
    # you can plot the prior distribution of mu with curve( dnorm( x , mu , sigma ) , from=x_start , to=x_end )
    # plot the prior distribution of the variable with
    # sample_mu <- rnorm( 1e4 , 178 , 20 )
    # sample_sigma <- runif( 1e4 , 0 , 50 )
    # prior_h <- rnorm( 1e4 , sample_mu , sample_sigma )
    # dens( prior_h )
    # extract samples from the model: 
    # post <- extract.samples( m1 )
    # post[1:5,]
    m1 <- NULL
    attempt <- 0
    while( is.null(m1) && attempt <= 3 ) {
      attempt <- attempt + 1
      try(
        # if it fails, it is still null, otherwise it is not null
        m1 <- map(
          alist(
            temp ~ dnorm(  mu , sigma ) ,
            mu <- a + b*month_index + Amp*sin(2*pi/12 * ( month_index + x)), # p. 113 # map2stan did not regognize pi, so I use 3.14159265359
            #a ~ dnorm( mean(d$temp) , 5 ) ,
            a ~ dnorm( a_prior_mean , 5) ,
            b ~ dnorm( 0 , 3) ,
            Amp ~ dnorm(0, 6),
            x ~ dnorm(0 , 6),
            sigma ~ dunif( 0 , 10 )
          ) ,
          data=d ) #,
        # start=list(a=a_prior_mean))
        
      )
    } 
    
    print(precis( m1 , digits = 5, corr=TRUE))
    
    mu <- link( m1 , data=data.frame(month_index=d$month_index) )
    
    mu.mean <- apply( mu , 2 , mean )
    mu.HPDI <- apply( mu , 2 , HPDI , prob=0.89 )
    
    sim.temp <- sim( m1 , data=list(month_index=d$month_index), n=1e4 ) # increasing number of samples smooths out the intervals
    str(sim.temp)
    
    temp.PI <- apply( sim.temp , 2 , PI , prob=0.89 )
    
    # calculate residuals
    muca <- coef(m1)['a'] + coef(m1)['b']*d$month_index + coef(m1)['Amp']*sin(2*pi/12 * ( d$month_index + coef(m1)['x']))
    # compute residual for each State (in percent)
    
    m.resid <- (d$temp - muca) # how to properly normalize?
    
    # plot everything
    
    
    # plot text (and save to .csv for latex Table)
    su1 <- summary(m1)
    # extract with su1@output$Mean[1] (1 for a, 2 for b, 3 for sigma)
    bstr1 <- su1@output$Mean[2]
    stdstr1 <- su1@output$StdDev[2]
    bstr55 <- su1@output['5.5%'][2,1]
    bstr945 <- su1@output['94.5%'][2,1]
    
    # *12 to have it per year
    bstr1.2 <- formatC(bstr1 * 12, digits=3,format="fg", flag="#")
    stdstr1.2 <- formatC(stdstr1 * 12, digits=2,format="fg", flag="#")
    bstr55.2 <- formatC(bstr55 * 12, digits=3,format="fg", flag="#")
    bstr945.2 <- formatC(bstr945 * 12, digits=3,format="fg", flag="#")
    
    plot( temp ~ month_index , data=d , col=col.alpha(rangi2,0.8), ylim= c(mean(d$temp)-offset, mean(d$temp)+offset),  cex=1, pch=19 , ylab = ylabel,font.main = 1, main=paste(csites[idx], forc, cpressure[idx] , sep=' ')) #
    mtext(paste('b = ', bstr1.2," \u00B1 ",stdstr1.2, ' ', unit, '/year\n89%-Interval = (', bstr55.2, ', ',bstr945.2,') ' ,unit ,'/year', sep=''), 
          side=3, line=-2, cex=0.7)
    
    lines( d$month_index , mu.mean )
    # plot a shaded region for 89% HPDI (highest posterior densitiy interval)
    shade( mu.HPDI  ,d$month_index )
    # draw PI (percentile intervals) region for simulated temp values
    shade( temp.PI , d$month_index )
    
    plot(d$month_index, m.resid)
    
    # write output to .csv
    write.csv(su1@output,  paste(base_path, csites[idx],'_', forc, cpressure[idx], '_Temperature_PRIMAVERA_Projections_Bayesian_model.csv' , sep=''), row.names=T)
    
  }
  dev.off() # close file
}
}
