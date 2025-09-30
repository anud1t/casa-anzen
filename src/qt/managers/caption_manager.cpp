#include "caption_manager.hpp"
#include <QNetworkRequest>
#include <QFile>
#include <QDebug>
#include <QFileInfo>
#include <QJsonDocument>
#include <QJsonObject>

CaptionManager::CaptionManager(QObject* parent)
    : QObject(parent)
    , m_networkManager(new QNetworkAccessManager(this))
    , m_captionEndpoint("http://localhost:2020/v1/caption")
{
    connect(m_networkManager, &QNetworkAccessManager::finished,
            this, &CaptionManager::onCaptionReplyFinished);
}

void CaptionManager::requestCaption(QListWidgetItem* item, const QString& imagePath)
{
    // Minimal logging by default
    
    if (!item || imagePath.isEmpty()) {
        qDebug() << "Caption failed: Invalid item or image path";
        emit captionFailed(item, "Invalid item or image path");
        return;
    }

    // FORCE OLD REQUEST: send data URL in image_url (as old implementation did)
    QFileInfo fi(imagePath);
    const QString absPath = fi.absoluteFilePath();
    if (!fi.exists()) {
        qDebug() << "Caption failed: File does not exist" << absPath;
        emit captionFailed(item, "File does not exist");
        return;
    }

    // Build data URL (JPEG) exactly like before
    const QString dataUrl = createDataUrl(absPath);
    if (dataUrl.isEmpty()) {
        qDebug() << "Caption failed: Failed to build data URL for" << absPath;
        emit captionFailed(item, "Failed to build data URL");
        return;
    }

    // Minimal request log

    QJsonObject payload;
    payload.insert("image_url", dataUrl);
    payload.insert("length", "short");
    
    QJsonDocument doc(payload);
    QByteArray body = doc.toJson(QJsonDocument::Compact);
    // Log truncated body for diagnostics (avoid spewing entire base64)
    QByteArray preview = body.left(180);
    // qDebug() << "Caption payload (trunc):" << preview << "..."; // disable noisy payload logging

    QNetworkRequest request{QUrl(m_captionEndpoint)};
    request.setHeader(QNetworkRequest::ContentTypeHeader, "application/json");

    QNetworkReply* reply = m_networkManager->post(request, body);
    m_pendingRequests[reply] = qMakePair(item, absPath);
    setupTimeoutFor(reply, 30000);
}

void CaptionManager::setCaptionEndpoint(const QString& endpoint)
{
    m_captionEndpoint = endpoint;
}

QString CaptionManager::createDataUrl(const QString& imagePath)
{
    QImage image(imagePath);
    if (image.isNull()) {
        return QString();
    }

    QByteArray bytes;
    QBuffer buffer(&bytes);
    buffer.open(QIODevice::WriteOnly);
    
    if (!image.save(&buffer, "JPEG")) {
        return QString();
    }

    QByteArray base64 = bytes.toBase64();
    return QStringLiteral("data:image/jpeg;base64,") + QString::fromLatin1(base64);
}

QString CaptionManager::createBase64(const QString& imagePath)
{
    QImage image(imagePath);
    if (image.isNull()) {
        return QString();
    }

    QByteArray bytes;
    QBuffer buffer(&bytes);
    buffer.open(QIODevice::WriteOnly);
    if (!image.save(&buffer, "JPEG")) {
        return QString();
    }
    return QString::fromLatin1(bytes.toBase64());
}

QString CaptionManager::extractCaptionFromResponse(const QByteArray& response)
{
    QJsonParseError error;
    QJsonDocument doc = QJsonDocument::fromJson(response, &error);
    
    if (error.error != QJsonParseError::NoError || !doc.isObject()) {
        return QString();
    }

    QJsonObject obj = doc.object();
    QString caption;
    
    // Accept several shapes
    if (obj.contains("caption")) {
        caption = obj.value("caption").toString();
    }
    if (caption.isEmpty() && obj.contains("answer")) {
        caption = obj.value("answer").toString();
    }
    if (caption.isEmpty() && obj.contains("data") && obj.value("data").isObject()) {
        QJsonObject data = obj.value("data").toObject();
        if (data.contains("caption")) caption = data.value("caption").toString();
        if (caption.isEmpty() && data.contains("answer")) caption = data.value("answer").toString();
    }

    return caption.isEmpty() ? "(no caption)" : caption;
}

void CaptionManager::handleCaptionError(QListWidgetItem* item, const QString& error)
{
    qDebug() << "Caption error for item:" << error;
    emit captionFailed(item, error);
}

void CaptionManager::onCaptionReplyFinished(QNetworkReply* reply)
{
    if (!reply) return;

    auto pair = m_pendingRequests.take(reply);
    QListWidgetItem* item = pair.first;
    const QString originalAbsPath = pair.second;
    if (!item) {
        qDebug() << "Caption reply finished but no item found";
        reply->deleteLater();
        return;
    }

    // Minimal reply status

    if (reply->error() != QNetworkReply::NoError) {
        qDebug() << "Caption failed with network error:" << reply->errorString();
        handleCaptionError(item, "Network error: " + reply->errorString());
        if (m_replyTimeouts.contains(reply)) { m_replyTimeouts.take(reply)->deleteLater(); }
        reply->deleteLater();
        return;
    }

    QByteArray response = reply->readAll();
    // Response body is parsed; omit verbose dump
    QString caption = extractCaptionFromResponse(response);
    // Do not auto-retry; we want raw server response for diagnostics
    
    if (caption.isEmpty()) {
        qDebug() << "Caption failed: Failed to parse caption response";
        emit captionFailed(item, "Failed to parse caption response");
    } else {
        qDebug() << "Caption successful:" << caption;
        emit captionReady(item, caption);
    }

    if (m_replyTimeouts.contains(reply)) { m_replyTimeouts.take(reply)->deleteLater(); }
    reply->deleteLater();
}

void CaptionManager::cancelRequestsForItem(QListWidgetItem* item)
{
    // Abort any in-flight requests mapped to this item
    auto it = m_pendingRequests.begin();
    while (it != m_pendingRequests.end()) {
        if (it.value().first == item) {
            QNetworkReply* r = it.key();
            if (m_replyTimeouts.contains(r)) { m_replyTimeouts.take(r)->deleteLater(); }
            if (r) {
                r->abort();
                r->deleteLater();
            }
            it = m_pendingRequests.erase(it);
        } else {
            ++it;
        }
    }
}

void CaptionManager::setupTimeoutFor(QNetworkReply* reply, int ms)
{
    QTimer* t = new QTimer(this);
    t->setSingleShot(true);
    connect(t, &QTimer::timeout, this, [this, reply]() {
        auto pair = m_pendingRequests.take(reply);
        QListWidgetItem* item = pair.first;
        if (item) {
            qDebug() << "Caption request timed out";
            handleCaptionError(item, "Caption request timed out");
        }
        if (reply) reply->abort();
    });
    m_replyTimeouts.insert(reply, t);
    t->start(ms);
}
