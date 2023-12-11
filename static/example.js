const FPS = 20;

const inputCanvas = document.createElement('canvas');
const inputContext = inputCanvas.getContext('2d');

const output = document.querySelector('h1');
const direction = document.querySelector('h2');

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

    video.srcObject = stream;

    socket.addEventListener('message', async (event) => {
        const outputData = event.data;
        text = await outputData.text();
        if (text.slice(0, 1) == "0") {
            output.innerHTML = text.slice(1);
        } else if (text.slice(0, 1) == "1") {
            direction.innerHTML = text.slice(1);
        }
    });

    setInterval(async () => {
        inputContext.drawImage(video, 0, 0);
        const blob = await getBlob();
        socket.send(blob);
    }, 1000 / FPS);
}

main();
