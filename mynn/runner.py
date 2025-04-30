import numpy as np
import os
from tqdm import tqdm

class RunnerM():
    """
    This is an exmaple to train, evaluate, save, load the model. However, some of the function calling may not be correct 
    due to the different implementation of those models.
    """
    def __init__(self, model, optimizer, metric, loss_fn, batch_size=32, scheduler=None):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.metric = metric
        self.scheduler = scheduler
        self.batch_size = batch_size

        self.train_scores = []
        self.dev_scores = []
        self.train_loss = []
        self.dev_loss = []

    def train(self, train_set, dev_set, **kwargs):

        num_epochs = kwargs.get("num_epochs", 0)
        log_iters = kwargs.get("log_iters", 100)
        save_dir = kwargs.get("save_dir", "best_model")

        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        best_score = 0

        for epoch in range(num_epochs):
            X, y = train_set

            assert X.shape[0] == y.shape[0]

            idx = np.random.permutation(range(X.shape[0]))

            X = X[idx]
            y = y[idx]

            for iteration in tqdm(range(int(X.shape[0] / self.batch_size) + 1)):
                train_X = X[iteration * self.batch_size : (iteration+1) * self.batch_size]
                train_y = y[iteration * self.batch_size : (iteration+1) * self.batch_size]

                logits = self.model(train_X)
                trn_loss = self.loss_fn(logits, train_y)
                self.train_loss.append(trn_loss)
                
                trn_score = self.metric(logits, train_y)
                self.train_scores.append(trn_score)

                # the loss_fn layer will propagate the gradients.
                self.loss_fn.backward()
                self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()
                
                if (iteration) % log_iters == 0:
                    self.model.eval()
                    dev_score, dev_loss = self.evaluate(dev_set)
                    self.model.train()
                    self.dev_scores.append(dev_score)
                    self.dev_loss.append(dev_loss)
                    print(f"epoch: {epoch}, iteration: {iteration}")
                    print(f"[Train] loss: {trn_loss}, score: {trn_score}")
                    print(f"[Dev] loss: {dev_loss}, score: {dev_score}")

                    if dev_score > best_score:
                        save_path = os.path.join(save_dir, 'best_model.pickle')
                        self.save_model(save_path)
                        print(f"best accuracy performence has been updated: {best_score:.5f} --> {dev_score:.5f}")
                        best_score = dev_score
                    self.best_score = best_score

    def evaluate(self, data_set):
        """
        评估模型性能，按批次处理数据
        
        Args:
            data_set: 包含 (X, y) 的元组，X 是输入数据，y 是标签
            
        Returns:
            tuple: (平均得分, 平均损失)
        """
        X, y = data_set
        n_samples = X.shape[0]
        total_loss = 0.0
        total_score = 0.0
        n_batches = int(np.ceil(n_samples / self.batch_size))
        
        for batch_idx in tqdm(range(n_batches)):
            # 获取当前批次的数据
            start_idx = batch_idx * self.batch_size
            end_idx = min((batch_idx + 1) * self.batch_size, n_samples)
            batch_X = X[start_idx:end_idx]
            batch_y = y[start_idx:end_idx]
            
            # 模型前向传播
            batch_logits = self.model(batch_X)
            
            # 计算损失和评分
            batch_loss = self.loss_fn(batch_logits, batch_y)
            batch_score = self.metric(batch_logits, batch_y)
            
            # 根据批次大小加权累加
            batch_weight = (end_idx - start_idx) / n_samples
            total_loss += batch_loss * batch_weight
            total_score += batch_score * batch_weight
        
        return total_score, total_loss
    
    def save_model(self, save_path):
        self.model.save_model(save_path)