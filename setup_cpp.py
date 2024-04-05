from zipfile import ZipFile
import os

if __name__ == "__main__":
    files = sorted([file for file in os.listdir('dist') 
             if file.startswith('hetgpy') and file.endswith('.whl')]
    )
    folder = f"dist/{files[-1]}" # take the last one -- most recent version
    
    cpp_names = ('gauss','matern','EMSE','qEI')
    zip = ZipFile(folder)
    zipfolder = [z for z in zip.namelist() if z.endswith('.so') and 'src' not in z]
    foo=1
    for z in zipfolder:
        name = f"{folder}/{z}"
        zip.extract(z)
        print(name)