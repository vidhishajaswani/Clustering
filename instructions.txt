1. Results kept in vjaswan.pdf
2. Keep vjaswan.csv, vjaswan.py, gmm.R in the same folder.
3. Run python file using the command (Written in Spyder IDE)

	python vjaswan.py

If any library does not exist simply use pip install <library_name> command to install that library. E.g. pip install seaboard

4. After running the file, the results are shown in the command prompt and the plots open up in a different window. The 3d plots can be rotated in those windows as well.

5. Otherwise, run python file from Spyder and simply use run button.
6. The Gausian Mixture Decomposition is done in the R file since it was much easier to implement in R than in Python. Install RStudio and to run file, first set the working directory as the present directory by going in
	
	Session->Set working Directory-> Browse

Open the gmm.R file in this as well.

To run the file use the run button. After running, the console will ask for one of the options out of BIC, Classification, Uncertainty, and density curves. Use options for density and classification to see the density and  contour plots.

