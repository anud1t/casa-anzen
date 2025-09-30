#pragma once

#include <QWidget>
#include <QListWidget>
#include <QPushButton>
#include <QLabel>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QNetworkAccessManager>

class EventCard;

class EventFeedWidget : public QWidget
{
    Q_OBJECT

public:
    explicit EventFeedWidget(QWidget* parent = nullptr);
    ~EventFeedWidget() = default;

    void addEvent(const QString& title, const QPixmap& thumbnail, const QString& caption = QString());
    void clearEvents();
    void setCaptionForItem(QListWidgetItem* item, const QString& caption);

signals:
    void viewRequested(QListWidgetItem* item);
    void captionRequested(QListWidgetItem* item);
    void deleteRequested(QListWidgetItem* item);
    void deleteAllRequested();

private slots:
    void onViewClicked();
    void onCaptionClicked();
    void onDeleteClicked();
    void onDeleteAllClicked();
    void onEventCardClicked();

private:
    void setupUI();
    void applyMilitaryTheme();
    void setupEventToolbar();
    QListWidgetItem* createEventItem(const QString& title, const QPixmap& thumbnail, const QString& caption);

    QVBoxLayout* m_mainLayout;
    QLabel* m_header;
    QWidget* m_toolbar;
    QListWidget* m_eventList;
    
    QPushButton* m_viewBtn;
    QPushButton* m_captionBtn;
    QPushButton* m_deleteBtn;
    QPushButton* m_deleteAllBtn;
    
    QNetworkAccessManager* m_networkManager;
};
