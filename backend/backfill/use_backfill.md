### Backfill foldeR:

Contains all script to backfill the DB's and get the data from forex-factory and mt5.

1. -> If you use the mt5 applications: set the candles on unlimited.

- Open mt5 program.
- Go to tools.
- Click on options.
- graphs -> Max bars in graph = unlimited.

2. Now you can run the script : backfill_historic_data.
3. For the events: just run the fackit.py
4. When the data is updated and in your DB.
5. Run indicator_pipeline : run_mode = full. (add arg to terminal. )
   - This can take a while. ( 18years of data will take around 6 to 8 hours. )
   - My pc stats: 5090rtx + 4070 super ti (lol) -> 7900x3d with 128ram (running on 3200mhz :( )) - using my D-drive 990 nvme samsung.
   - Biggest bottleneck will be the events. (Due to the gentle request preventing the IP-blockage.)
     - Bottleneck after that will the be indicator_pipeline. (evals)
