#pragma once

#include <QObject>
#include <QString>
#include <QPixmap>
#include <QListWidgetItem>
#include <QDir>
#include <QFile>

class EventFeedWidget;
class CaptionManager;

class EventManager : public QObject
{
    Q_OBJECT

public:
    explicit EventManager(QObject* parent = nullptr);
    ~EventManager() = default;

    void setEventFeed(EventFeedWidget* eventFeed);
    void setCaptionManager(CaptionManager* captionManager);
    
    void addEvent(const QString& title, const QPixmap& thumbnail, const QString& caption = QString());
    void deleteEvent(QListWidgetItem* item);
    void deleteAllEvents();
    void clearEvents();
    
    void requestCaption(QListWidgetItem* item, const QString& imagePath);
    void viewEvent(QListWidgetItem* item);
    
    QString getCaptureDirPath() const;
    QString getCaptureSubdirPath(const QString& sub) const;

signals:
    void eventAdded(const QString& title);
    void eventDeleted(const QString& title);
    void allEventsDeleted();
    void captionReady(QListWidgetItem* item, const QString& caption);
    void captionFailed(QListWidgetItem* item, const QString& error);

private slots:
    void onCaptionReady(QListWidgetItem* item, const QString& caption);
    void onCaptionFailed(QListWidgetItem* item, const QString& error);

private:
    void setupEventHandling();
    QString captureDirPath() const;
    QString captureSubdirPath(const QString& sub) const;
    void ensureCaptureDirectories();

    EventFeedWidget* m_eventFeed;
    CaptionManager* m_captionManager;
    
    QString m_captureBasePath;
    QDir m_captureDir;
};
