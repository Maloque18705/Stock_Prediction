import torch
import numpy as np
import os
from data.build import process
from src.model import PriceModel
from src.optimizer import Optimizer
from src.train import Trainer
from src.plot import Plotter

def main():
    ticker = "TCB" # ["VCB, CTG, BID, TCB, MBB, ACB"]
    start_date = "2023-01-01"
    end_date = "2025-03-01"
 
    sequence_length = 3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    processor = process(ticker)
    df = processor.fetch_data(start_date, end_date)
    df = processor.data_scaling(df, fit=not os.path.exists(processor.scaler_path))
    
    valid_start = df.index[sequence_length]
    valid_end = df.index[-1]
    if df.empty:
        print("No data")
        return
    
    window_df = processor.df_to_windowed_df(df, valid_start, valid_end, n=sequence_length)
    if window_df.empty:
        print("Create window failed")
        return
    
    dates, X, y = processor.window_df_to_date_X_y(window_df)

    splits = processor.split_data(dates, X, y)
    if splits is None:
        print("Split data failed")
        return
    
    _, X_train, y_train = splits['train']
    _, X_test, y_test = splits['test']

    model = PriceModel(input_size=1)
    optimizer_setup = Optimizer(model, X_train, y_train, batch_size=16, learning_rate=1e-3, device=device)


    trainer = Trainer(
        model=optimizer_setup.model,
        optimizer=optimizer_setup.optimizer,
        loss_fn=optimizer_setup.loss_fn,
        loader=optimizer_setup.loader,
        X_train_tensor=optimizer_setup.X_train_tensor,
        y_train_tensor=optimizer_setup.y_train_tensor,
        X_test=X_test,
        y_test=y_test,
        device=device,
    )

    trainer.train(n_epochs=8000, eval_every=100)
    
    os.makedirs('./saved_model', exist_ok=True)
    torch.save(model.state_dict(), './saved_model/model.pth')

    plotter = Plotter(model,optimizer_setup.X_train_tensor, y_train, trainer.X_test_tensor, y_test, device)
    plotter.plot()

if __name__ == "__main__":
    main()
