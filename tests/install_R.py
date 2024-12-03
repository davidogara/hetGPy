from rpy2.robjects import r; 

if __name__ == "__main__":
    r("""
      dir.create(Sys.getenv("R_LIBS_USER"), recursive = TRUE)  # create personal library
      .libPaths(Sys.getenv("R_LIBS_USER"))  # add to the path
      deps <-c('hetGP','R.matlab','plgp')
      to_install <- deps[!(deps %in% installed.packages()[,"Package"])]
      if(length(to_install)) install.packages(to_install,repos='https://cloud.r-project.org')
      """)