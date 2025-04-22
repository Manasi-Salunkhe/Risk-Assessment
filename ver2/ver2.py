# Import Required Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import joblib

# Deep Learning Libraries - PyTorch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# Streamlit UI Setup
st.set_page_config(page_title='Risk Assessment', layout='wide')
st.title('📊 Deep Learning Risk Assessment System')

# Load Dataset or Upload
uploaded_file = st.file_uploader("Upload Project Dataset (CSV)", type=["csv"])
if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.subheader("📄 Dataset Preview")
    st.dataframe(data.head())

    # Preprocessing
    st.subheader("🔍 Data Preprocessing")
    label_encoders = {}
    if 'Risk_Level' not in data.columns:
        st.error("Dataset must contain 'Risk_Level' column for training.")
        st.stop()

    for col in data.select_dtypes(include=['object']).columns:
        if col == 'Risk_Level':
            # Ensure Risk_Level is ordinal: Low < Medium < High
            risk_mapping = {'Low': 0, 'Medium': 1, 'High': 2}
            data[col] = data[col].map(risk_mapping)
        else:
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col])
            label_encoders[col] = le

    X = data.drop('Risk_Level', axis=1)
    y = data['Risk_Level']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Model Selection
    st.subheader("🧠 Model Selection")
    model_category = st.radio("Select Model Category", ["Machine Learning", "Deep Learning"])
    
    if model_category == "Machine Learning":
        model_option = st.selectbox("Choose ML Model",
                                ["Random Forest", "Decision Tree", "Logistic Regression", "Naive Bayes", "SVM"])
        
        # ML Model Selection
        if model_option == "Random Forest":
            model = RandomForestClassifier()
        elif model_option == "Decision Tree":
            model = DecisionTreeClassifier()
        elif model_option == "Logistic Regression":
            model = LogisticRegression(max_iter=1000)
        elif model_option == "Naive Bayes":
            model = GaussianNB()
        elif model_option == "SVM":
            model = SVC(probability=True)
        
        # Train ML Model
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_probs = model.predict_proba(X_test)
        
        st.subheader("✅ Evaluation Metrics")
        st.text(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
        st.text("\nClassification Report:")
        st.text(classification_report(y_test, y_pred))
        
        # For feature importance (ML models only)
        has_feature_importance = hasattr(model, 'feature_importances_')
        
    else:  # Deep Learning with PyTorch
        input_dim = X_train.shape[1]
        dl_model_option = st.selectbox("Choose Deep Learning Model", 
                                     ["Simple Neural Network", "Deep Neural Network", "Advanced Neural Network"])
        
        # Deep Learning Hyperparameters
        col1, col2, col3 = st.columns(3)
        with col1:
            epochs = st.slider("Epochs", min_value=10, max_value=200, value=50, step=10)
        with col2:
            batch_size = st.slider("Batch Size", min_value=8, max_value=128, value=32, step=8)
        with col3:
            learning_rate = st.select_slider("Learning Rate", 
                                           options=[0.0001, 0.001, 0.01, 0.1], 
                                           value=0.001)

        # Define PyTorch Neural Network Models
        class SimpleNN(nn.Module):
            def __init__(self, input_dim):
                super(SimpleNN, self).__init__()
                self.fc1 = nn.Linear(input_dim, 16)
                self.fc2 = nn.Linear(16, 8)
                self.fc3 = nn.Linear(8, 3)
                self.relu = nn.ReLU()
                
            def forward(self, x):
                x = self.relu(self.fc1(x))
                x = self.relu(self.fc2(x))
                x = self.fc3(x)
                return x
                
        class DeepNN(nn.Module):
            def __init__(self, input_dim):
                super(DeepNN, self).__init__()
                self.fc1 = nn.Linear(input_dim, 32)
                self.bn1 = nn.BatchNorm1d(32)
                self.fc2 = nn.Linear(32, 16)
                self.fc3 = nn.Linear(16, 8)
                self.fc4 = nn.Linear(8, 3)
                self.relu = nn.ReLU()
                self.dropout = nn.Dropout(0.3)
                
            def forward(self, x):
                x = self.relu(self.bn1(self.fc1(x)))
                x = self.dropout(x)
                x = self.relu(self.fc2(x))
                x = self.relu(self.fc3(x))
                x = self.fc4(x)
                return x
                
        class AdvancedNN(nn.Module):
            def __init__(self, input_dim):
                super(AdvancedNN, self).__init__()
                self.fc1 = nn.Linear(input_dim, 64)
                self.bn1 = nn.BatchNorm1d(64)
                self.fc2 = nn.Linear(64, 32)
                self.bn2 = nn.BatchNorm1d(32)
                self.fc3 = nn.Linear(32, 16)
                self.bn3 = nn.BatchNorm1d(16)
                self.fc4 = nn.Linear(16, 8)
                self.fc5 = nn.Linear(8, 3)
                self.relu = nn.ReLU()
                self.dropout = nn.Dropout(0.3)
                
            def forward(self, x):
                x = self.relu(self.bn1(self.fc1(x)))
                x = self.dropout(x)
                x = self.relu(self.bn2(self.fc2(x)))
                x = self.dropout(x)
                x = self.relu(self.bn3(self.fc3(x)))
                x = self.relu(self.fc4(x))
                x = self.fc5(x)
                return x
        
        # Select the neural network architecture
        if dl_model_option == "Simple Neural Network":
            net = SimpleNN(input_dim)
        elif dl_model_option == "Deep Neural Network":
            net = DeepNN(input_dim)
        else:  # Advanced Neural Network
            net = AdvancedNN(input_dim)
            
        # Display network architecture
        st.text("Neural Network Architecture:")
        st.code(str(net))
        
        # Convert data to PyTorch tensors
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.LongTensor(y_train.values)
        X_test_tensor = torch.FloatTensor(X_test)
        
        # Create data loaders
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # Loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(net.parameters(), lr=learning_rate)
        
        # Training progress trackers
        train_losses = []
        train_accs = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Training loop
        for epoch in range(epochs):
            net.train()
            running_loss = 0.0
            correct = 0
            total = 0
            
            for inputs, labels in train_loader:
                optimizer.zero_grad()
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            
            epoch_loss = running_loss / len(train_loader)
            epoch_acc = 100 * correct / total
            
            train_losses.append(epoch_loss)
            train_accs.append(epoch_acc)
            
            # Update progress
            progress = (epoch + 1) / epochs
            progress_bar.progress(progress)
            status_text.text(f"Training Progress: {int(progress*100)}% (Epoch {epoch+1}/{epochs})")
        
        # Clear progress display after training
        progress_bar.empty()
        status_text.empty()
        
        # Plot training history
        fig_history, ax_history = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot training loss
        ax_history[0].plot(train_losses)
        ax_history[0].set_title('Model Loss')
        ax_history[0].set_ylabel('Loss')
        ax_history[0].set_xlabel('Epoch')
        
        # Plot training accuracy
        ax_history[1].plot(train_accs)
        ax_history[1].set_title('Model Accuracy')
        ax_history[1].set_ylabel('Accuracy (%)')
        ax_history[1].set_xlabel('Epoch')
        
        st.pyplot(fig_history)
        
        # Evaluate the model
        net.eval()
        with torch.no_grad():
            outputs = net(X_test_tensor)
            _, y_pred = torch.max(outputs, 1)
            y_pred = y_pred.numpy()
            
            # Get probabilities using softmax
            probs = nn.functional.softmax(outputs, dim=1).numpy()
            y_probs = probs  # For consistency with ML models
        
        st.subheader("✅ Evaluation Metrics")
        st.text(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
        st.text("\nClassification Report:")
        st.text(classification_report(y_test, y_pred))
        
        # Store model for later use
        model = net
        
        # No feature importance for neural networks
        has_feature_importance = False

    # Risk Analysis Visualization
    st.subheader("📊 Risk Analysis Dashboard")
    
    # Create tabs for different visualizations
    tab1, tab2, tab3 = st.tabs(["Risk Distribution", "Feature Importance", "Probability Analysis"])
    
    with tab1:
        st.subheader("Project Risk Distribution")
        risk_counts = pd.Series(y_pred).value_counts().reindex(index=[0, 1, 2], fill_value=0).values
        risk_labels = ['Low', 'Medium', 'High']
        
        fig1, ax1 = plt.subplots(figsize=(8, 6))
        bars = ax1.bar(risk_labels, risk_counts, 
                      color=['#4CAF50', '#FFC107', '#F44336'])
        ax1.set_title('Predicted Risk Levels Distribution', fontsize=14)
        ax1.set_ylabel('Number of Projects')
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}',
                    ha='center', va='bottom')
        
        st.pyplot(fig1)
    
    with tab2:
        if has_feature_importance:
            st.subheader("Feature Importance Analysis")
            importance = pd.DataFrame({
                'Feature': X.columns,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            fig2, ax2 = plt.subplots(figsize=(10, 6))
            sns.barplot(x='Importance', y='Feature', data=importance, palette='viridis', ax=ax2)
            ax2.set_title('Feature Importance for Risk Prediction', fontsize=14)
            st.pyplot(fig2)
        else:
            if model_category == "Deep Learning":
                st.info("Feature importance visualization is not available for neural network models. Consider using techniques like SHAP for neural network interpretability.")
            else:
                st.info("Feature importance not available for this model type")
    
    with tab3:
        st.subheader("Risk Probability Distribution")
        
        if model_category == "Machine Learning":
            prob_df = pd.DataFrame(y_probs, columns=['Low', 'Medium', 'High'])
        else:  # Deep Learning
            prob_df = pd.DataFrame(y_probs, columns=['Low', 'Medium', 'High'])
        
        fig3, ax3 = plt.subplots(figsize=(10, 6))
        sns.boxplot(data=prob_df, palette=['#4CAF50', '#FFC107', '#F44336'], ax=ax3)
        ax3.set_title('Probability Distribution Across Risk Levels', fontsize=14)
        ax3.set_ylabel('Probability')
        ax3.set_xlabel('Risk Level')
        st.pyplot(fig3)

    # Risk Prediction for New Input
    st.subheader("🔮 Predict Risk Level for New Project")
    input_data = {}
    for feature in X.columns:
        input_data[feature] = st.number_input(f"Enter value for {feature}", min_value=0.0)

    if st.button("Predict Risk"):
        input_df = pd.DataFrame([input_data])
        input_scaled = scaler.transform(input_df)
        
        if model_category == "Machine Learning":
            prediction = model.predict(input_scaled)
            prob = model.predict_proba(input_scaled)[0]
        else:  # Deep Learning with PyTorch
            input_tensor = torch.FloatTensor(input_scaled)
            model.eval()
            with torch.no_grad():
                output = model(input_tensor)
                _, prediction = torch.max(output, 1)
                prob = nn.functional.softmax(output, dim=1).numpy()[0]
                prediction = prediction.numpy()
        
        # Display prediction with color-coded risk level
        risk_labels = ['Low', 'Medium', 'High']
        colors = ['#4CAF50', '#FFC107', '#F44336']
        
        st.markdown(
            f"<h3 style='text-align: center; color:{colors[prediction[0]]};'>"
            f"Predicted Risk Level: {risk_labels[prediction[0]]}</h3>",
            unsafe_allow_html=True
        )
        
        # Create a styled probability display
        prob_df = pd.DataFrame({
            'Risk Level': risk_labels,
            'Probability': prob
        })
        
        # Display probability gauge chart
        fig4, ax4 = plt.subplots(figsize=(10, 2))
        ax4.barh(['Risk Probability'], [1], color='lightgray')
        ax4.barh(['Risk Probability'], [prob[prediction[0]]], 
                color=colors[prediction[0]])
        ax4.set_xlim(0, 1)
        ax4.set_title('Prediction Confidence', fontsize=12)
        ax4.text(0.5, 0, f"{prob[prediction[0]]:.1%}", 
                ha='center', va='center', color='white', fontsize=14)
        st.pyplot(fig4)
        
        # Display full probability breakdown
        st.dataframe(prob_df.style.format({'Probability': '{:.2%}'}).background_gradient(
            cmap='YlOrRd', subset=['Probability']))

    # Save Model
    if st.button("💾 Save Trained Model"):
        if model_category == "Machine Learning":
            joblib.dump(model, 'risk_model.pkl')
            st.success("ML Model saved as 'risk_model.pkl'")
        else:  # Deep Learning with PyTorch
            torch.save(model.state_dict(), 'risk_model_dl.pth')
            st.success("Deep Learning Model saved as 'risk_model_dl.pth'")

    # Download Report
    st.subheader("📄 Download Report")
    report = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    csv = report_df.to_csv(index=True).encode('utf-8')
    st.download_button("Download CSV Report", csv, "report.csv", "text/csv")

else:
    st.warning("Please upload a CSV file to start analysis.")
    
    # Sample Data Generator (Optional helper for users without data)
    st.subheader("Don't have a dataset? Generate a sample one")
    if st.button("Generate Sample Dataset"):
        # Create a sample dataset for software projects
        np.random.seed(42)
        n_samples = 200
        
        # Generate features that might be relevant for software project risk assessment
        team_size = np.random.randint(3, 30, n_samples)
        budget = np.random.randint(10000, 500000, n_samples)
        duration_months = np.random.randint(1, 36, n_samples)
        complexity_score = np.random.uniform(1, 10, n_samples)
        stakeholder_count = np.random.randint(1, 20, n_samples)
        requirements_stability = np.random.uniform(0, 1, n_samples)
        team_experience = np.random.uniform(1, 10, n_samples)
        
        # Create DataFrame
        sample_data = pd.DataFrame({
            'TeamSize': team_size,
            'Budget': budget,
            'DurationMonths': duration_months,
            'ComplexityScore': complexity_score,
            'StakeholderCount': stakeholder_count,
            'RequirementsStability': requirements_stability,
            'TeamExperience': team_experience
        })
        
        # Generate risk levels based on a simple formula
        risk_score = (
            (10 - team_experience) * 0.3 + 
            complexity_score * 0.25 + 
            (1 - requirements_stability) * 0.25 + 
            (stakeholder_count / 20) * 0.1 + 
            (duration_months / 36) * 0.1
        )
        
        # Assign risk levels based on the calculated risk score
        risk_level = pd.cut(
            risk_score, 
            bins=[0, 3.33, 6.66, 10], 
            labels=['Low', 'Medium', 'High']
        )
        
        sample_data['Risk_Level'] = risk_level
        
        # Save sample data as CSV and offer download
        csv = sample_data.to_csv(index=False).encode('utf-8')
        st.download_button(
            "Download Sample Dataset", 
            csv, 
            "sample_project_risk_data.csv", 
            "text/csv"
        )
        
        st.success("Sample dataset generated! Click the button above to download it.")
        st.dataframe(sample_data.head())