#include "event_manager.hpp"
#include "../views/event_feed_widget.hpp"
#include "caption_manager.hpp"
#include <QDebug>
#include <QCoreApplication>
#include <QProcess>

EventManager::EventManager(QObject* parent)
    : QObject(parent)
    , m_eventFeed(nullptr)
    , m_captionManager(nullptr)
{
    setupEventHandling();
    ensureCaptureDirectories();
}

void EventManager::setEventFeed(EventFeedWidget* eventFeed)
{
    m_eventFeed = eventFeed;
}

void EventManager::setCaptionManager(CaptionManager* captionManager)
{
    m_captionManager = captionManager;
    
    if (m_captionManager) {
        connect(m_captionManager, &CaptionManager::captionReady,
                this, &EventManager::onCaptionReady);
        connect(m_captionManager, &CaptionManager::captionFailed,
                this, &EventManager::onCaptionFailed);
    }
}

void EventManager::addEvent(const QString& title, const QPixmap& thumbnail, const QString& caption)
{
    if (!m_eventFeed) {
        qDebug() << "Event feed not set, cannot add event";
        return;
    }

    m_eventFeed->addEvent(title, thumbnail, caption);
    emit eventAdded(title);
    
    qDebug() << "Event added:" << title;
}

void EventManager::deleteEvent(QListWidgetItem* item)
{
    if (!item || !m_eventFeed) {
        return;
    }

    // Get event path from item data if available
    QString path = item->data(Qt::UserRole).toString();
    if (!path.isEmpty()) {
        QFile file(path);
        if (file.exists()) {
            if (!file.remove()) {
                qDebug() << "Failed to delete file:" << path;
            }
        }
    }

    // Remove from UI
    m_eventFeed->clearEvents(); // This will be handled by the event feed
    
    emit eventDeleted("Event");
    qDebug() << "Event deleted";
}

void EventManager::deleteAllEvents()
{
    if (!m_eventFeed) {
        return;
    }

    // Remove files under data/captures recursively
    QDir capRoot(captureDirPath());
    if (capRoot.exists()) {
        capRoot.removeRecursively();
        // Recreate base dir to avoid issues elsewhere
        QDir().mkpath(captureDirPath());
    }

    // Clear UI
    m_eventFeed->clearEvents();
    
    emit allEventsDeleted();
    qDebug() << "All events deleted";
}

void EventManager::clearEvents()
{
    if (m_eventFeed) {
        m_eventFeed->clearEvents();
    }
}

void EventManager::requestCaption(QListWidgetItem* item, const QString& imagePath)
{
    if (!m_captionManager) {
        qDebug() << "Caption manager not set, cannot request caption";
        return;
    }

    m_captionManager->requestCaption(item, imagePath);
}

void EventManager::viewEvent(QListWidgetItem* item)
{
    if (!item) {
        return;
    }

    QString path = item->data(Qt::UserRole).toString();
    if (!path.isEmpty()) {
        // Open file with default application
        QProcess::startDetached("xdg-open", QStringList() << path);
    }
}

QString EventManager::getCaptureDirPath() const
{
    return captureDirPath();
}

QString EventManager::getCaptureSubdirPath(const QString& sub) const
{
    return captureSubdirPath(sub);
}

void EventManager::onCaptionReady(QListWidgetItem* item, const QString& caption)
{
    if (m_eventFeed) {
        m_eventFeed->setCaptionForItem(item, caption);
    }
    emit captionReady(item, caption);
}

void EventManager::onCaptionFailed(QListWidgetItem* item, const QString& error)
{
    qDebug() << "Caption failed for item:" << error;
    emit captionFailed(item, error);
}

void EventManager::setupEventHandling()
{
    // Setup any additional event handling logic
}

QString EventManager::captureDirPath() const
{
    // Resolve to project root's data/captures: executable is in build/, so go up one
    QDir appDir(QCoreApplication::applicationDirPath());
    appDir.cdUp();
    QString path = appDir.filePath("data/captures");
    return path;
}

QString EventManager::captureSubdirPath(const QString& sub) const
{
    QDir root(captureDirPath());
    return root.filePath(sub);
}

void EventManager::ensureCaptureDirectories()
{
    QString basePath = captureDirPath();
    QDir().mkpath(basePath);
    QDir().mkpath(captureSubdirPath("persons"));
    QDir().mkpath(captureSubdirPath("vehicles"));
}
