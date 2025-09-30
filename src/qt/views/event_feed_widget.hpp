#pragma once

#include <QWidget>
#include <QListWidget>
#include <QPushButton>
#include <QLabel>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QNetworkAccessManager>
#include <QTimer>

class EventCard;

class EventFeedWidget : public QWidget
{
    Q_OBJECT

public:
    explicit EventFeedWidget(QWidget* parent = nullptr);
    ~EventFeedWidget() = default;

    void addEvent(const QString& title, const QPixmap& thumbnail, const QString& caption = QString(), const QString& imagePath = QString());
    void clearEvents();
    void setCaptionForItem(QListWidgetItem* item, const QString& caption);
    // Expose list for controlled removals
    QListWidget* getList() const { return m_eventList; }
    
    // Defensive helper: safely remove a specific item
    void removeItem(QListWidgetItem* item) {
        if (!m_eventList || !item) return;
        int row = m_eventList->row(item);
        if (row >= 0) {
            QListWidgetItem* taken = m_eventList->takeItem(row);
            // Defer deletion to avoid use-after-free during selection change signals
            QTimer::singleShot(0, m_eventList, [taken]() { delete taken; });
        }
    }
    QListWidgetItem* getLastItem() const;

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
    void onSelectionChanged();

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
