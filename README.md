# Movies-ETL

Using try-except blocks to account for unforeseen problems that may arise with new data.

## Assumption 1 - correct loading of data for successful import of data:
- files are imported correctly,
- file format is correct,
- file directory is correct

## Assumption 2 - converting numeric columns:
-	Converting from object to int, if the value is not a number it will create an error

## Assumption 3 - converting to datetime:
-	Time stamp and release date are in the expected format

## Assumption 4 - correct password in a config file:
-	Correct password in a config file, without it no connection to the database

## Assumption 5 - import of the data:
-	Assumption is that database exist and is available
