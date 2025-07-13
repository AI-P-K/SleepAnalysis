1. Extract the main clues and information from the task sheet
2. Familiarize with the proposed architectures 
    - RNN 
    - LSTM (fixed the problem of long term dependencies in RNN) do we have large sequences in our dataset?
      - granularity of data ?
      - how many time steps per sample ?
      - Try GRU as is faster then LSTM and can provide same accuracy depending on the dataset
3. Create a new anaconda environment
4. Familiarize with the dataset
   - Why negative values for timestamps?
   - Different lengths in the dataframes why?
   - How to match this dataframes (raw_data_processor.py, chatGPT, paper)
   - Different separators in the CSVs  (',' and ' ')
   - Two different types of matching the data 1hz signals (merge each second to the nearest label within ±15s tolerance) or window a 30-second sequence and assign the label based on the final timestamp or majority vote
   - How magnitude influences training ? It measures  how much movement happened, not only in what direction
5. Create a first prototype with data processing, architecture, training, validation 
   - for a single patient (check if pipeline works for a single patient)
   - for all patients (extend to all patients) create train, val splits and a first view of accuracy
6. Review the pipeline, modularize and production ready code (logs)
7. Add options for multiple types of architectures such as RNN, LSTM, GRU and Transformers
8. Add different options for data alignment (30 seconds as used in https://github.com/ojwalch/sleep_classifiers/) or 1s for higher granularity 
9. Add option to add another feature to the dataset motion magnitude so we can test how this influences the performance
10. Option to remove negative values in the dataset
11. The training script is built dynamically - it adapts to features set and is suited for multiple hyperparameter testing
12. Make each run traceable with timestamped folders, saving hyperparameters, data processing error logs, models, scaler, plots, accuracies csvs
13. Serve the inference scripy with FastAPI no async needed unless we scale it for DB or I/O use
14. Containerize the application using Docker