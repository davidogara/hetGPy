'''
Utility script to handle installing R packages (required for more advanced tests)

'''
if __name__ == "__main__":
    from rpy2.robjects import r;  
    r("""
      dir.create(Sys.getenv("R_LIBS_USER"), recursive = TRUE)  # create personal library
      .libPaths(Sys.getenv("R_LIBS_USER"))  # add to the path
      deps <-c('hetGP','R.matlab','plgp','yaml')
      to_install <- deps[!(deps %in% installed.packages()[,"Package"])]
      if(length(to_install)) install.packages(to_install,repos='https://cloud.r-project.org')
      """)