"""
Test price prediction model
"""
import numpy as np
from trading_bot.ml.model import PricePredictor

def main():
    # Create config
    config = {
        'sequence_length': 10,
        'n_features': 5,
        'batch_size': 32,
        'epochs': 2,
        'hidden_units': 64
    }

    # Create dummy data
    print("Creating dummy data...")
    X_train = np.random.random((100, 10, 5))
    y_train = {
        'direction': np.eye(2)[np.random.randint(0, 2, 100)],
        'magnitude': np.random.random((100, 1)),
        'confidence': np.random.randint(0, 2, (100, 1))
    }

    X_val = np.random.random((20, 10, 5))
    y_val = {
        'direction': np.eye(2)[np.random.randint(0, 2, 20)],
        'magnitude': np.random.random((20, 1)),
        'confidence': np.random.randint(0, 2, (20, 1))
    }

    try:
        # Create model
        print("Creating model...")
        model = PricePredictor(config)
        
        # Print model summary
        print("\nModel summary:")
        print(model.get_summary())
        
        # Train model
        print("\nTraining model...")
        model.fit(
            X_train, 
            y_train,
            X_val=X_val,
            y_val=y_val,
            epochs=2
        )
        
        # Make predictions
        print("\nMaking predictions...")
        direction, magnitude, confidence = model.predict(X_val)
        
        print(f"\nPrediction shapes:")
        print(f"Direction shape: {direction.shape}")
        print(f"Magnitude shape: {magnitude.shape}")
        print(f"Confidence shape: {confidence.shape}")
        
        # Evaluate model
        print("\nEvaluating model...")
        metrics = model.evaluate(X_val, y_val)
        print("Metrics:", metrics)
        
        print("\nTest completed successfully!")
        
    except Exception as e:
        print(f"Error during test: {e}")
        raise

if __name__ == "__main__":
    main()
