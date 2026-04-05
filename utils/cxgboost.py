import json
import joblib
from typing import Dict, Optional
import xgboost
import numpy as np
from sklearn.ensemble import RandomForestClassifier


class CXGBoost:
    """
    Causal XGBoost model that jointly estimates potential outcomes Y(0) and Y(1)
    using DragonNet-style treatment masking.

    Supports both binary and multiclass classification tasks.
    """

    def __init__(
        self, 
        parameters: Optional[dict] = None, 
        propensity_model=None,
        task: str = 'multiclass',
        loss_type: str = 'mse',
        scale_pos_weight: Optional[float] = None,
    ) -> None:
        """
        Initialize the CXGBoost model.
        
        Args:
            parameters: XGBoost hyperparameters (dict)
            propensity_model: Sklearn-style classifier for propensity scores
            task: Type of classification task - one of:
                - 'binary': Binary classification (uses XGBClassifier)
                - 'multiclass': Multiclass classification (uses XGBRegressor)
            loss_type: Type of loss function:
                For 'binary': 'mse', 'bce', 'weighted_bce', 'huber', 'focal'
                For 'multiclass': automatically set to 'mse'
            scale_pos_weight: Weight for positive class in weighted_bce (binary only)
        """
        self.task = task
        self.parameters = parameters or {}
        
        # Validate task and set up model accordingly
        if self.task == 'binary':
            self.loss_type = loss_type
            self.scale_pos_weight = scale_pos_weight
            self.model = xgboost.XGBClassifier(**self.parameters)
        elif self.task == 'multiclass':            
            # Force MSE loss for multiclass
            self.loss_type = 'mse'
            self.scale_pos_weight = None
            self.model = xgboost.XGBRegressor(**self.parameters)            
        else:
            raise ValueError(f"Unknown task: {task}. Must be 'binary' or 'multiclass'")
        
        # Initialize propensity score model
        self.propensity_model = propensity_model or RandomForestClassifier(
            n_estimators=100,
            max_depth=3,
            class_weight='balanced',
            n_jobs=-1,
            random_state=42,
        )

    def fit(self, X: np.ndarray, t: np.ndarray, y: np.ndarray) -> None:
        """
        Fit the causal model and the propensity score model.
        
        Args:
            X: Covariates, shape (n_samples, n_features)
            t: Treatment indicator, shape (n_samples,), binary 0/1
            y: Outcome, shape (n_samples,)
                - For binary task: binary 0/1
                - For multiclass task: integer class labels 0, 1, ..., num_classes-1
        """
        
        # Validate input
        if self.task == 'multiclass':
            self.unique_classes = np.unique(y)

        # Create treatment indicator mask (DragonNet style)
        tt = np.array([[1, 0] if t_i == 0 else [0, 1] for t_i in t]).flatten()
        
        # Auto-compute scale_pos_weight for binary task if needed
        if self.task == 'binary' and self.loss_type == 'weighted_bce' and self.scale_pos_weight is None:
            n_pos = np.sum(y == 1)
            n_neg = np.sum(y == 0)
            computed_weight = n_neg / n_pos if n_pos > 0 else 1.0
            print(f"Auto-computed scale_pos_weight: {computed_weight:.3f}")
            self.scale_pos_weight = computed_weight
        
        # Define the custom loss function based on task and loss_type
        if self.task == 'binary':
            custom_loss = self._get_binary_loss(tt)
        else:  # multiclass
            custom_loss = self._get_multiclass_loss(tt)
        
        # Configure XGBoost
        self.model.set_params(
            objective=custom_loss,
            num_target=2,
            multi_strategy="multi_output_tree"
        )
        
        # Prepare multi-output target: [y, y] for both Y(0) and Y(1)
        yt = np.column_stack([y, y])
        
        # Fit the XGBoost model
        print(f"Fitting CXGBoost ({self.task}) with {self.loss_type} loss...")
        print(f"Treatment distribution: {np.mean(t):.2%} treated, {1-np.mean(t):.2%} control")
        self.model.fit(X, yt)
        
        # Fit the propensity score model
        print("Fitting propensity model...")
        self.propensity_model.fit(X, t)
    
    def _get_binary_loss(self, tt: np.ndarray):
        """Get loss function for binary classification task."""
        
        if self.loss_type == 'weighted_bce':
            scale = self.scale_pos_weight
            
            def custom_loss(y_true: np.ndarray, y_pred: np.ndarray) -> tuple:
                """Weighted binary cross-entropy with treatment masking."""
                preds = 1.0 / (1.0 + np.exp(-y_pred.flatten()))
                preds = np.clip(preds, 1e-7, 1 - 1e-7)
                
                y_flat = y_true.flatten()
                weights = np.where(y_flat == 1, scale, 1.0)
                
                grad = (preds - y_flat) * weights * tt
                hess = preds * (1.0 - preds) * weights * tt
                hess = np.maximum(hess, 1e-16)
                
                return grad, hess
        
        elif self.loss_type == 'bce':
            def custom_loss(y_true: np.ndarray, y_pred: np.ndarray) -> tuple:
                """Binary cross-entropy with treatment masking."""
                preds = 1.0 / (1.0 + np.exp(-y_pred.flatten()))
                preds = np.clip(preds, 1e-7, 1 - 1e-7)
                
                grad = (preds - y_true.flatten()) * tt
                hess = preds * (1.0 - preds) * tt
                hess = np.maximum(hess, 1e-16)
                
                return grad, hess
        
        elif self.loss_type == 'focal':
            def custom_loss(y_true: np.ndarray, y_pred: np.ndarray, 
                          alpha: float = 0.25, gamma: float = 2.0) -> tuple:
                """Focal loss with treatment masking."""
                preds = 1.0 / (1.0 + np.exp(-y_pred.flatten()))
                preds = np.clip(preds, 1e-7, 1 - 1e-7)
                
                y = y_true.flatten()
                pt = np.where(y == 1, preds, 1 - preds)
                focal_weight = alpha * (1 - pt) ** gamma
                
                grad = focal_weight * (preds - y) * tt
                hess = focal_weight * preds * (1 - preds) * tt
                hess = np.maximum(hess, 1e-16)
                
                return grad, hess
        
        elif self.loss_type == 'huber':
            def custom_loss(y_true: np.ndarray, y_pred: np.ndarray, 
                          delta: float = 1.0) -> tuple:
                """Huber loss with treatment masking."""
                residual = y_pred.flatten() - y_true.flatten()
                abs_residual = np.abs(residual)
                
                grad = np.where(
                    abs_residual <= delta,
                    residual,
                    delta * np.sign(residual)
                ) * tt
                
                hess = np.where(
                    abs_residual <= delta,
                    1.0,
                    delta / (abs_residual + 1e-8)
                ) * tt
                hess = np.maximum(hess, 1e-16)
                
                return grad, hess
        
        elif self.loss_type == 'mse':
            def custom_loss(y_true: np.ndarray, y_pred: np.ndarray) -> tuple:
                """Mean squared error with treatment masking."""
                grad = 2 * (y_pred.flatten() - y_true.flatten()) * tt
                hess = np.full_like(y_pred.flatten(), 2.0) * tt
                hess = np.maximum(hess, 1e-16)
                
                return grad, hess
        
        else:
            raise ValueError(
                f"Unknown loss_type for binary task: {self.loss_type}. "
                f"Must be one of: 'mse', 'bce', 'weighted_bce', 'huber', 'focal'"
            )
        
        return custom_loss
    
    def _get_multiclass_loss(self, tt: np.ndarray):
        """Get MSE loss for multiclass classification task."""
        
        def custom_loss(y_true: np.ndarray, y_pred: np.ndarray) -> tuple:
            """Mean squared error with treatment masking for multiclass."""
            grad = 2 * (y_pred.flatten() - y_true.flatten()) * tt
            hess = np.full_like(y_pred.flatten(), 2.0) * tt
            hess = np.maximum(hess, 1e-16)
            
            return grad, hess
        
        return custom_loss

    def predict(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Generate potential outcome predictions and propensity scores.
        
        Args:
            X: Covariates, shape (n_samples, n_features)
            
        Returns:
            Dictionary with:
                - 'y_0_hat': Predicted outcome under control (T=0)
                    Binary: probabilities in [0, 1]
                    Multiclass: class probabilities, shape (n_samples, num_classes)
                - 'y_1_hat': Predicted outcome under treatment (T=1)
                    Binary: probabilities in [0, 1]
                    Multiclass: class probabilities, shape (n_samples, num_classes)
                - 'propensity_score': P(T=1|X)
        """
        if self.task == 'binary':
            pred = self.model.predict_proba(X)
        else:
            pred = self.model.predict(X)
        y_0_hat = pred[:, 0]  # Y_0 prediction
        y_1_hat = pred[:, 1]  # Y_1 prediction
        
        if self.task == 'multiclass':
            y_0_hat = np.where(y_0_hat < self.unique_classes[0], self.unique_classes[0], y_0_hat)
            y_0_hat = np.where(y_0_hat > self.unique_classes[-1], self.unique_classes[-1], y_0_hat)
            y_1_hat = np.where(y_1_hat < self.unique_classes[0], self.unique_classes[0], y_1_hat)
            y_1_hat = np.where(y_1_hat > self.unique_classes[-1], self.unique_classes[-1], y_1_hat)
            
        return {
            'y_0_hat': y_0_hat,
            'y_1_hat': y_1_hat,
            'propensity_score': self.propensity_model.predict_proba(X)[:, 1]
        }
    def save(self, path_prefix: str) -> None:
        """
        Save the model to disk.
        
        Args:
            path_prefix: Path prefix for saved files, e.g. 'models/my_model'
                         Will create: <path_prefix>_xgb.ubj
                                      <path_prefix>_propensity.pkl
                                      <path_prefix>_meta.json
        """
        self.model.set_params(objective=None)
        self.model.save_model(f'{path_prefix}_xgb.ubj')

        joblib.dump(self.propensity_model, f'{path_prefix}_propensity.pkl')

        meta = {
            'task': self.task,
            'loss_type': self.loss_type,
            'parameters': self.parameters,
            'unique_classes': self.unique_classes.tolist() if self.task == 'multiclass' else None,
            'scale_pos_weight': self.scale_pos_weight,
        }
        with open(f'{path_prefix}_meta.json', 'w') as f:
            json.dump(meta, f, indent=2)
        
        print(f"Model saved to {path_prefix}_{{xgb.ubj, propensity.pkl, meta.json}}")

    @classmethod
    def load(cls, path_prefix: str) -> 'CXGBoost':
        """
        Load a saved model from disk.
        
        Args:
            path_prefix: Same path prefix used when saving
            
        Returns:
            Loaded CXGBoost instance ready for prediction
        """
        with open(f'{path_prefix}_meta.json', 'r') as f:
            meta = json.load(f)

        instance = cls(parameters=meta['parameters'], task=meta['task'])
        instance.model.load_model(f'{path_prefix}_xgb.ubj')
        instance.propensity_model = joblib.load(f'{path_prefix}_propensity.pkl')
        instance.scale_pos_weight = meta['scale_pos_weight']
        
        if meta['unique_classes'] is not None:
            instance.unique_classes = np.array(meta['unique_classes'])

        return instance        