const FPS = 5;

const inputCanvas = document.createElement('canvas');
const inputContext = inputCanvas.getContext('2d');

const outputCanvas = document.querySelector('canvas');
const outputContext = outputCanvas.getContext('2d');

const video = document.querySelector('video');

const socket = new WebSocket('ws://127.0.0.1:5000/socket');

async function getBlob() {
    return new Promise((resolve, reject) => {
        try {
            inputCanvas.toBlob(resolve, 'image/png');
        } catch (error) {
            reject(error);
        }
    });
}

async function main() {
    const stream = await navigator.mediaDevices.getUserMedia({ video: true });
    const tracks = stream.getVideoTracks();
    const { width, height } = tracks[0].getSettings();

    inputCanvas.width = width;
    inputCanvas.height = height;

    outputCanvas.width = width;
    outputCanvas.height = height;

    video.srcObject = stream;

    socket.addEventListener('message', async (event) => {
        const outputData = event.data;
        const bitmap = await createImageBitmap(outputData);
        outputContext.drawImage(bitmap, 0, 0);
    });

    setInterval(async () => {
        inputContext.drawImage(video, 0, 0);
        const blob = await getBlob();
        socket.send(blob);
    }, 1000 / FPS);
}

main();
