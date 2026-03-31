import sys
import os
from datetime import datetime
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTextEdit, QPushButton, QLabel, QProgressBar, QFileDialog,
    QTableWidget, QTableWidgetItem, QTabWidget, QGroupBox,
    QMessageBox, QStatusBar, QComboBox, QSpinBox, QDoubleSpinBox,
    QCheckBox, QSplitter
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QFont, QColor, QPalette, QIcon
import re
import json


class ModelWorker(QThread):
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)
    progress = pyqtSignal(int)
    
    def __init__(self, text, model, tokenizer, device, max_length=512):
        super().__init__()
        self.text = text
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_length = max_length
        
    def run(self):
        try:
            # Предобработка текста
            self.progress.emit(25)
            processed_text = self.preprocess_text(self.text)
            
            # Токенизация
            self.progress.emit(50)
            inputs = self.tokenizer(
                processed_text,
                return_tensors="pt",
                max_length=self.max_length,
                padding="max_length",
                truncation=True
            )
            
            # Перемещаем на устройство
            inputs = {key: val.to(self.device) for key, val in inputs.items()}
            
            # Инференс
            self.progress.emit(75)
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probabilities = torch.nn.functional.softmax(logits, dim=-1)
                predicted_class = torch.argmax(probabilities, dim=-1).item()
                confidence = probabilities[0][predicted_class].item()
            
            self.progress.emit(100)
            
            # Формируем результат
            result = {
                'class_id': predicted_class,
                'confidence': confidence,
                'probabilities': probabilities[0].cpu().numpy().tolist(),
                'text': self.text,
                'processed_text': processed_text,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            self.finished.emit(result)
            
        except Exception as e:
            self.error.emit(str(e))
    
    def preprocess_text(self, text):
        """Предобработка текста"""
        # Удаление URL
        text = re.sub(r'http\S+|www.\S+', '[URL]', text)
        # Удаление лишних пробелов
        text = re.sub(r'\s+', ' ', text)
        # Удаление переносов строк
        text = text.replace('\n', ' ').replace('\r', ' ')
        return text.strip()


class ContentAnalyzerGUI(QMainWindow):
    """Главное окно приложения"""
    
    # Определение классов контента
    CLASSES = {
        0: {'name': 'Насилие', 'color': '#ff4444', 'description': 'Призывы к физическому насилию, угрозы'},
        1: {'name': 'Ненависть', 'color': '#ff9944', 'description': 'Оскорбления, дискриминация'},
        2: {'name': 'Суицид', 'color': '#aa44ff', 'description': 'Пропаганда или обсуждение самоубийства'},
        3: {'name': 'Дезинформация', 'color': '#4477ff', 'description': 'Намеренно ложная информация'},
        4: {'name': 'Нейтральный', 'color': '#44ff44', 'description': 'Безопасный контент'}
    }
    
    def __init__(self):
        super().__init__()
        self.model = None
        self.tokenizer = None
        self.device = None
        self.history = []
        self.current_worker = None
        
        self.init_ui()
        self.setup_styles()
        
    def init_ui(self):
        """Инициализация пользовательского интерфейса"""
        self.setWindowTitle("Система выявления опасного контента - ruBERT Analyzer")
        self.setGeometry(100, 100, 1200, 800)
        
        # Центральный виджет
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # Создаем вкладки
        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs)
        
        # Вкладка анализа
        self.create_analysis_tab()
        
        # Вкладка истории
        self.create_history_tab()
        
        # Вкладка настроек
        self.create_settings_tab()
        
        # Вкладка информации
        self.create_info_tab()
        
        # Статус бар
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        self.statusBar.showMessage("Готов к работе")
        
    def create_analysis_tab(self):
        """Создание вкладки анализа текста"""
        analysis_widget = QWidget()
        layout = QVBoxLayout(analysis_widget)
        
        # Заголовок
        title = QLabel("🔍 Анализ текстового контента")
        title.setFont(QFont("Arial", 16, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        
        # Группа загрузки модели
        model_group = QGroupBox("Загрузка модели")
        model_layout = QHBoxLayout()
        
        self.model_path_label = QLabel("Модель не загружена")
        self.model_path_label.setStyleSheet("color: #ff4444;")
        model_layout.addWidget(self.model_path_label)
        
        self.load_model_btn = QPushButton("📁 Загрузить модель")
        self.load_model_btn.clicked.connect(self.load_model)
        model_layout.addWidget(self.load_model_btn)
        
        self.device_label = QLabel("Устройство: CPU")
        model_layout.addWidget(self.device_label)
        
        model_group.setLayout(model_layout)
        layout.addWidget(model_group)
        
        # Сплиттер для разделения ввода и результатов
        splitter = QSplitter(Qt.Horizontal)
        
        # Левая панель - ввод текста
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        
        input_label = QLabel("Введите текст для анализа:")
        input_label.setFont(QFont("Arial", 11, QFont.Bold))
        left_layout.addWidget(input_label)
        
        self.input_text = QTextEdit()
        self.input_text.setPlaceholderText(
            "Введите или вставьте текст для анализа...\n\n"
            "Примеры:\n"
            "- Текстовые сообщения\n"
            "- Комментарии из социальных сетей\n"
            "- Посты с форумов\n"
            "- Любой другой текстовый контент"
        )
        self.input_text.setMinimumHeight(200)
        left_layout.addWidget(self.input_text)
        
        # Кнопки действий
        action_layout = QHBoxLayout()
        
        self.analyze_btn = QPushButton("🚀 Анализировать")
        self.analyze_btn.clicked.connect(self.analyze_text)
        self.analyze_btn.setEnabled(False)
        self.analyze_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-size: 14px;
                font-weight: bold;
                padding: 10px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
        """)
        action_layout.addWidget(self.analyze_btn)
        
        self.clear_btn = QPushButton("🗑️ Очистить")
        self.clear_btn.clicked.connect(self.clear_input)
        action_layout.addWidget(self.clear_btn)
        
        self.load_file_btn = QPushButton("📄 Загрузить из файла")
        self.load_file_btn.clicked.connect(self.load_text_file)
        action_layout.addWidget(self.load_file_btn)
        
        left_layout.addLayout(action_layout)
        
        # Прогресс бар
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        left_layout.addWidget(self.progress_bar)
        
        splitter.addWidget(left_panel)
        
        # Правая панель - результаты
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        result_label = QLabel("Результаты анализа:")
        result_label.setFont(QFont("Arial", 11, QFont.Bold))
        right_layout.addWidget(result_label)
        
        # Основной результат
        self.result_display = QGroupBox("Классификация")
        result_display_layout = QVBoxLayout()
        
        self.class_label = QLabel("Класс: -")
        self.class_label.setFont(QFont("Arial", 14, QFont.Bold))
        result_display_layout.addWidget(self.class_label)
        
        self.confidence_label = QLabel("Уверенность: -")
        self.confidence_label.setFont(QFont("Arial", 12))
        result_display_layout.addWidget(self.confidence_label)
        
        self.description_label = QLabel("")
        self.description_label.setWordWrap(True)
        result_display_layout.addWidget(self.description_label)
        
        self.result_display.setLayout(result_display_layout)
        right_layout.addWidget(self.result_display)
        
        # Детальные вероятности
        prob_group = QGroupBox("Детальные вероятности по классам")
        prob_layout = QVBoxLayout()
        
        self.probability_labels = {}
        for class_id, class_info in self.CLASSES.items():
            label = QLabel(f"{class_info['name']}: 0.00%")
            label.setStyleSheet(f"padding: 5px; border-left: 4px solid {class_info['color']};")
            self.probability_labels[class_id] = label
            prob_layout.addWidget(label)
        
        prob_group.setLayout(prob_layout)
        right_layout.addWidget(prob_group)
        
        # Кнопки действий с результатами
        result_action_layout = QHBoxLayout()
        
        self.save_result_btn = QPushButton("💾 Сохранить результат")
        self.save_result_btn.clicked.connect(self.save_current_result)
        self.save_result_btn.setEnabled(False)
        result_action_layout.addWidget(self.save_result_btn)
        
        self.copy_result_btn = QPushButton("📋 Копировать")
        self.copy_result_btn.clicked.connect(self.copy_result)
        self.copy_result_btn.setEnabled(False)
        result_action_layout.addWidget(self.copy_result_btn)
        
        right_layout.addLayout(result_action_layout)
        
        splitter.addWidget(right_panel)
        splitter.setSizes([600, 600])
        
        layout.addWidget(splitter)
        
        self.tabs.addTab(analysis_widget, "🔍 Анализ")
    
    def create_history_tab(self):
        """Создание вкладки истории анализов"""
        history_widget = QWidget()
        layout = QVBoxLayout(history_widget)
        
        title = QLabel("📊 История анализов")
        title.setFont(QFont("Arial", 16, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        
        # Таблица истории
        self.history_table = QTableWidget()
        self.history_table.setColumnCount(5)
        self.history_table.setHorizontalHeaderLabels([
            "Время", "Текст (фрагмент)", "Класс", "Уверенность", "Действия"
        ])
        self.history_table.horizontalHeader().setStretchLastSection(False)
        self.history_table.setColumnWidth(0, 150)
        self.history_table.setColumnWidth(1, 400)
        self.history_table.setColumnWidth(2, 150)
        self.history_table.setColumnWidth(3, 100)
        self.history_table.setColumnWidth(4, 150)
        layout.addWidget(self.history_table)
        
        # Кнопки управления историей
        history_action_layout = QHBoxLayout()
        
        self.export_history_btn = QPushButton("📤 Экспорт истории")
        self.export_history_btn.clicked.connect(self.export_history)
        history_action_layout.addWidget(self.export_history_btn)
        
        self.clear_history_btn = QPushButton("🗑️ Очистить историю")
        self.clear_history_btn.clicked.connect(self.clear_history)
        history_action_layout.addWidget(self.clear_history_btn)
        
        history_action_layout.addStretch()
        
        self.history_count_label = QLabel("Записей: 0")
        history_action_layout.addWidget(self.history_count_label)
        
        layout.addLayout(history_action_layout)
        
        self.tabs.addTab(history_widget, "📊 История")
    
    def create_settings_tab(self):
        """Создание вкладки настроек"""
        settings_widget = QWidget()
        layout = QVBoxLayout(settings_widget)
        
        title = QLabel("⚙️ Настройки")
        title.setFont(QFont("Arial", 16, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        
        # Настройки модели
        model_settings_group = QGroupBox("Параметры модели")
        model_settings_layout = QVBoxLayout()
        
        # Max length
        max_length_layout = QHBoxLayout()
        max_length_layout.addWidget(QLabel("Максимальная длина токенов:"))
        self.max_length_spin = QSpinBox()
        self.max_length_spin.setRange(128, 1024)
        self.max_length_spin.setValue(512)
        self.max_length_spin.setSingleStep(64)
        max_length_layout.addWidget(self.max_length_spin)
        max_length_layout.addStretch()
        model_settings_layout.addLayout(max_length_layout)
        
        # Порог уверенности
        threshold_layout = QHBoxLayout()
        threshold_layout.addWidget(QLabel("Порог уверенности:"))
        self.threshold_spin = QDoubleSpinBox()
        self.threshold_spin.setRange(0.0, 1.0)
        self.threshold_spin.setValue(0.7)
        self.threshold_spin.setSingleStep(0.05)
        self.threshold_spin.setDecimals(2)
        threshold_layout.addWidget(self.threshold_spin)
        threshold_layout.addStretch()
        model_settings_layout.addLayout(threshold_layout)
        
        # Устройство
        device_layout = QHBoxLayout()
        device_layout.addWidget(QLabel("Устройство вычислений:"))
        self.device_combo = QComboBox()
        self.device_combo.addItems(["CPU", "CUDA (GPU)"])
        if not torch.cuda.is_available():
            self.device_combo.setCurrentIndex(0)
            self.device_combo.setEnabled(False)
        device_layout.addWidget(self.device_combo)
        device_layout.addStretch()
        model_settings_layout.addLayout(device_layout)
        
        model_settings_group.setLayout(model_settings_layout)
        layout.addWidget(model_settings_group)
        
        # Настройки отображения
        display_settings_group = QGroupBox("Настройки отображения")
        display_settings_layout = QVBoxLayout()
        
        self.auto_scroll_check = QCheckBox("Автоматическая прокрутка к результатам")
        self.auto_scroll_check.setChecked(True)
        display_settings_layout.addWidget(self.auto_scroll_check)
        
        self.show_probabilities_check = QCheckBox("Показывать все вероятности")
        self.show_probabilities_check.setChecked(True)
        display_settings_layout.addWidget(self.show_probabilities_check)
        
        self.save_to_history_check = QCheckBox("Автоматически сохранять в историю")
        self.save_to_history_check.setChecked(True)
        display_settings_layout.addWidget(self.save_to_history_check)
        
        display_settings_group.setLayout(display_settings_layout)
        layout.addWidget(display_settings_group)
        
        layout.addStretch()
        
        # Кнопка применения настроек
        apply_btn = QPushButton("✅ Применить настройки")
        apply_btn.clicked.connect(self.apply_settings)
        layout.addWidget(apply_btn)
        
        self.tabs.addTab(settings_widget, "⚙️ Настройки")
    
    def create_info_tab(self):
        """Создание информационной вкладки"""
        info_widget = QWidget()
        layout = QVBoxLayout(info_widget)
        
        title = QLabel("ℹ️ О системе")
        title.setFont(QFont("Arial", 16, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        
        info_text = QTextEdit()
        info_text.setReadOnly(True)
        info_text.setHtml("""
        <h2>Система выявления опасного контента</h2>
        
        <h3>Описание</h3>
        <p>Система автоматического выявления опасного контента в текстовых сообщениях 
        на основе дообученной трансформерной модели ruBERT для русского языка.</p>
        
        <h3>Возможности</h3>
        <ul>
            <li>Классификация текстов по 5 категориям</li>
            <li>Анализ единичных сообщений и диалогов</li>
            <li>Работа с русскоязычным контентом</li>
            <li>Локальное выполнение без отправки данных на внешние серверы</li>
            <li>История анализов с возможностью экспорта</li>
        </ul>
        
        <h3>Категории контента</h3>
        <ul>
            <li><b style="color: #ff4444;">Насилие</b> - Призывы к физическому насилию, угрозы</li>
            <li><b style="color: #ff9944;">Ненависть</b> - Оскорбления, дискриминация по различным признакам</li>
            <li><b style="color: #aa44ff;">Суицид</b> - Пропаганда или обсуждение способов самоубийства</li>
            <li><b style="color: #4477ff;">Дезинформация</b> - Намеренно ложная информация</li>
            <li><b style="color: #44ff44;">Нейтральный</b> - Безопасный контент</li>
        </ul>
        
        <h3>Технологии</h3>
        <ul>
            <li>Python 3.8+</li>
            <li>PyTorch</li>
            <li>Transformers (Hugging Face)</li>
            <li>ruBERT (предобученная модель для русского языка)</li>
            <li>PyQt5 (графический интерфейс)</li>
        </ul>
        
        <h3>Использование</h3>
        <ol>
            <li>Загрузите дообученную модель через кнопку "Загрузить модель"</li>
            <li>Введите или вставьте текст для анализа</li>
            <li>Нажмите "Анализировать" для получения результата</li>
            <li>Просмотрите детальные вероятности по всем классам</li>
            <li>Сохраните результат или просмотрите историю анализов</li>
        </ol>
        
        <h3>Производительность</h3>
        <p>Система обрабатывает тексты за 100-500 мс на обычном процессоре, 
        что позволяет использовать её для модерации контента в реальном времени.</p>
        
        <h3>Автор</h3>
        <p>Гребенюков Д.А., Студент группы КТбо4-12<br>
        Южный Федеральный Университет, 2025</p>
        
        <h3>Версия</h3>
        <p>1.0.0</p>
        """)
        layout.addWidget(info_text)
        
        self.tabs.addTab(info_widget, "ℹ️ О системе")
    
    def setup_styles(self):
        """Настройка стилей приложения"""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f5f5f5;
            }
            QGroupBox {
                font-weight: bold;
                border: 2px solid #cccccc;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
            QPushButton {
                padding: 8px;
                border-radius: 4px;
                background-color: #2196F3;
                color: white;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
            QPushButton:pressed {
                background-color: #0D47A1;
            }
            QTextEdit, QTableWidget {
                border: 1px solid #cccccc;
                border-radius: 4px;
                background-color: white;
            }
            QLabel {
                color: #333333;
            }
            QStatusBar {
                background-color: #e0e0e0;
                color: #333333;
            }
        """)
    
    def load_model(self):
        """Загрузка модели ruBERT"""
        model_path = QFileDialog.getExistingDirectory(
            self, "Выберите директорию с моделью"
        )
        
        if not model_path:
            return
        
        try:
            self.statusBar.showMessage("Загрузка модели...")
            QApplication.processEvents()
            
            # Определяем устройство
            if self.device_combo.currentText() == "CUDA (GPU)" and torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
            
            # Загружаем токенайзер
            self.tokenizer = BertTokenizer.from_pretrained(model_path)
            
            # Загружаем модель
            self.model = BertForSequenceClassification.from_pretrained(model_path)
            self.model.to(self.device)
            self.model.eval()
            
            # Обновляем интерфейс
            self.model_path_label.setText(f"Модель загружена: {os.path.basename(model_path)}")
            self.model_path_label.setStyleSheet("color: #44ff44;")
            self.device_label.setText(f"Устройство: {self.device.type.upper()}")
            self.analyze_btn.setEnabled(True)
            
            self.statusBar.showMessage("Модель успешно загружена", 3000)
            
            QMessageBox.information(
                self, "Успех",
                f"Модель успешно загружена!\nУстройство: {self.device.type.upper()}"
            )
            
        except Exception as e:
            self.statusBar.showMessage("Ошибка загрузки модели", 3000)
            QMessageBox.critical(
                self, "Ошибка",
                f"Не удалось загрузить модель:\n{str(e)}"
            )
    
    def analyze_text(self):
        """Анализ введенного текста"""
        text = self.input_text.toPlainText().strip()
        
        if not text:
            QMessageBox.warning(self, "Предупреждение", "Введите текст для анализа")
            return
        
        if not self.model or not self.tokenizer:
            QMessageBox.warning(self, "Предупреждение", "Сначала загрузите модель")
            return
        
        # Отключаем кнопку и показываем прогресс
        self.analyze_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.statusBar.showMessage("Анализ текста...")
        
        # Создаем и запускаем worker
        max_length = self.max_length_spin.value()
        self.current_worker = ModelWorker(text, self.model, self.tokenizer, self.device, max_length)
        self.current_worker.finished.connect(self.on_analysis_finished)
        self.current_worker.error.connect(self.on_analysis_error)
        self.current_worker.progress.connect(self.progress_bar.setValue)
        self.current_worker.start()
    
    def on_analysis_finished(self, result):
        """Обработка завершения анализа"""
        # Включаем кнопку и скрываем прогресс
        self.analyze_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.statusBar.showMessage("Анализ завершен", 3000)
        
        # Отображаем результаты
        self.display_results(result)
        
        # Сохраняем в историю
        if self.save_to_history_check.isChecked():
            self.history.append(result)
            self.update_history_table()
        
        # Активируем кнопки сохранения
        self.save_result_btn.setEnabled(True)
        self.copy_result_btn.setEnabled(True)
        
        # Сохраняем текущий результат
        self.current_result = result
    
    def on_analysis_error(self, error_msg):
        """Обработка ошибки анализа"""
        self.analyze_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.statusBar.showMessage("Ошибка анализа", 3000)
        
        QMessageBox.critical(
            self, "Ошибка анализа",
            f"Произошла ошибка при анализе текста:\n{error_msg}"
        )
    
    def display_results(self, result):
        """Отображение результатов анализа"""
        class_id = result['class_id']
        confidence = result['confidence']
        probabilities = result['probabilities']
        
        class_info = self.CLASSES[class_id]
        
        # Основной результат
        self.class_label.setText(f"Класс: {class_info['name']}")
        self.class_label.setStyleSheet(f"color: {class_info['color']};")
        
        self.confidence_label.setText(f"Уверенность: {confidence*100:.2f}%")
        
        threshold = self.threshold_spin.value()
        if confidence < threshold:
            warning = f"⚠️ Низкая уверенность (< {threshold*100:.0f}%). Рекомендуется ручная проверка."
            self.description_label.setText(f"{class_info['description']}\n\n{warning}")
            self.description_label.setStyleSheet("color: #ff9944;")
        else:
            self.description_label.setText(class_info['description'])
            self.description_label.setStyleSheet("color: #333333;")
        
        # Детальные вероятности
        if self.show_probabilities_check.isChecked():
            for idx, prob in enumerate(probabilities):
                class_name = self.CLASSES[idx]['name']
                self.probability_labels[idx].setText(f"{class_name}: {prob*100:.2f}%")
                
                # Подсветка максимальной вероятности
                if idx == class_id:
                    color = self.CLASSES[idx]['color']
                    self.probability_labels[idx].setStyleSheet(
                        f"padding: 5px; border-left: 4px solid {color}; "
                        f"background-color: {color}22; font-weight: bold;"
                    )
                else:
                    color = self.CLASSES[idx]['color']
                    self.probability_labels[idx].setStyleSheet(
                        f"padding: 5px; border-left: 4px solid {color};"
                    )
    
    def clear_input(self):
        """Очистка поля ввода"""
        self.input_text.clear()
    
    def load_text_file(self):
        """Загрузка текста из файла"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Выберите текстовый файл",
            "", "Text Files (*.txt);;All Files (*)"
        )
        
        if file_path:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                self.input_text.setPlainText(text)
                self.statusBar.showMessage(f"Загружен файл: {os.path.basename(file_path)}", 3000)
            except Exception as e:
                QMessageBox.critical(
                    self, "Ошибка",
                    f"Не удалось загрузить файл:\n{str(e)}"
                )
    
    def save_current_result(self):
        """Сохранение текущего результата"""
        if not hasattr(self, 'current_result'):
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Сохранить результат",
            f"result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            "JSON Files (*.json);;Text Files (*.txt)"
        )
        
        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(self.current_result, f, ensure_ascii=False, indent=2)
                self.statusBar.showMessage(f"Результат сохранен: {os.path.basename(file_path)}", 3000)
            except Exception as e:
                QMessageBox.critical(
                    self, "Ошибка",
                    f"Не удалось сохранить результат:\n{str(e)}"
                )
    
    def copy_result(self):
        """Копирование результата в буфер обмена"""
        if not hasattr(self, 'current_result'):
            return
        
        result = self.current_result
        class_info = self.CLASSES[result['class_id']]
        
        text = f"""Результат анализа текста
        
Класс: {class_info['name']}
Уверенность: {result['confidence']*100:.2f}%
Время: {result['timestamp']}

Текст:
{result['text']}

Детальные вероятности:
"""
        for idx, prob in enumerate(result['probabilities']):
            text += f"{self.CLASSES[idx]['name']}: {prob*100:.2f}%\n"
        
        clipboard = QApplication.clipboard()
        clipboard.setText(text)
        self.statusBar.showMessage("Результат скопирован в буфер обмена", 3000)
    
    def update_history_table(self):
        """Обновление таблицы истории"""
        self.history_table.setRowCount(len(self.history))
        
        for row, result in enumerate(reversed(self.history)):
            # Время
            time_item = QTableWidgetItem(result['timestamp'])
            self.history_table.setItem(row, 0, time_item)
            
            # Текст (фрагмент)
            text_fragment = result['text'][:50] + "..." if len(result['text']) > 50 else result['text']
            text_item = QTableWidgetItem(text_fragment)
            self.history_table.setItem(row, 1, text_item)
            
            # Класс
            class_info = self.CLASSES[result['class_id']]
            class_item = QTableWidgetItem(class_info['name'])
            class_item.setForeground(QColor(class_info['color']))
            self.history_table.setItem(row, 2, class_item)
            
            # Уверенность
            confidence_item = QTableWidgetItem(f"{result['confidence']*100:.2f}%")
            self.history_table.setItem(row, 3, confidence_item)
            
            # Действия
            actions_item = QTableWidgetItem("📋 Детали")
            self.history_table.setItem(row, 4, actions_item)
        
        self.history_count_label.setText(f"Записей: {len(self.history)}")
    
    def export_history(self):
        """Экспорт истории в файл"""
        if not self.history:
            QMessageBox.information(self, "Информация", "История пуста")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Экспорт истории",
            f"history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            "JSON Files (*.json);;CSV Files (*.csv)"
        )
        
        if file_path:
            try:
                if file_path.endswith('.json'):
                    with open(file_path, 'w', encoding='utf-8') as f:
                        json.dump(self.history, f, ensure_ascii=False, indent=2)
                elif file_path.endswith('.csv'):
                    import csv
                    with open(file_path, 'w', encoding='utf-8', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow(['Время', 'Текст', 'Класс', 'Уверенность'])
                        for result in self.history:
                            class_name = self.CLASSES[result['class_id']]['name']
                            writer.writerow([
                                result['timestamp'],
                                result['text'],
                                class_name,
                                f"{result['confidence']*100:.2f}%"
                            ])
                
                self.statusBar.showMessage(f"История экспортирована: {os.path.basename(file_path)}", 3000)
                QMessageBox.information(self, "Успех", "История успешно экспортирована")
                
            except Exception as e:
                QMessageBox.critical(
                    self, "Ошибка",
                    f"Не удалось экспортировать историю:\n{str(e)}"
                )
    
    def clear_history(self):
        """Очистка истории"""
        if not self.history:
            return
        
        reply = QMessageBox.question(
            self, "Подтверждение",
            "Вы уверены, что хотите очистить историю?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            self.history.clear()
            self.update_history_table()
            self.statusBar.showMessage("История очищена", 3000)
    
    def apply_settings(self):
        """Применение настроек"""
        self.statusBar.showMessage("Настройки применены", 3000)
        QMessageBox.information(self, "Успех", "Настройки успешно применены")


def main():
    """Точка входа в приложение"""
    app = QApplication(sys.argv)
    app.setApplicationName("ruBERT Content Analyzer")
    
    window = ContentAnalyzerGUI()
    window.show()
    
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()