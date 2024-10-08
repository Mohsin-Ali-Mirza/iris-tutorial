document.addEventListener('DOMContentLoaded', function() {
    const data = [
        [1, 5.1, 3.5, 1.4, 0.2, 'Iris-setosa'],
        [2, 4.9, 3.0, 1.4, 0.2, 'Iris-setosa'],
        [3, 4.7, 3.2, 1.3, 0.2, 'Iris-setosa'],
        [4,4.6,3.1,1.5,0.2,"Iris-setosa"],
        [5,5.0,3.6,1.4,0.2,"Iris-setosa"],
        [6,5.4,3.9,1.7,0.4,"Iris-setosa"],
        [7,4.6,3.4,1.4,0.3,"Iris-setosa"],
        [8,5.0,3.4,1.5,0.2,"Iris-setosa"],
        [9,4.4,2.9,1.4,0.2,"Iris-setosa"],
        [10,4.9,3.1,1.5,0.1,"Iris-setosa"],
        [11,5.4,3.7,1.5,0.2,"Iris-setosa"],
        [12,4.8,3.4,1.6,0.2,"Iris-setosa"],
        [13,4.8,3.0,1.4,0.1,"Iris-setosa"],
        [14,4.3,3.0,1.1,0.1,"Iris-setosa"],
        [15,5.8,4.0,1.2,0.2,"Iris-setosa"],
        [16,5.7,4.4,1.5,0.4,"Iris-setosa"],
        [17,5.4,3.9,1.3,0.4,"Iris-setosa"],
        [18,5.1,3.5,1.4,0.3,"Iris-setosa"],
        [19,5.7,3.8,1.7,0.3,"Iris-setosa"],
        [20,5.1,3.8,1.5,0.3,"Iris-setosa]"],
        [21,5.4,3.4,1.7,0.2,"Iris-setosa"],
        [22,5.1,3.7,1.5,0.4,"Iris-setosa"],
        [23,4.6,3.6,1.0,0.2,"Iris-setosa"],
        [24,5.1,3.3,1.7,0.5,"Iris-setosa"],
        [25,4.8,3.4,1.9,0.2,"Iris-setosa"],
        [26,5.0,3.0,1.6,0.2,"Iris-setosa"],
        [27,5.0,3.4,1.6,0.4,"Iris-setosa"],
        [28,5.2,3.5,1.5,0.2,"Iris-setosa"],
        [29,5.2,3.4,1.4,0.2,"Iris-setosa"],
        [30,4.7,3.2,1.6,0.2,"Iris-setosa"],
    ];

    const tbody = document.querySelector('table tbody');
    data.forEach(row => {
        const tr = document.createElement('tr');
        row.forEach(cell => {
            const td = document.createElement('td');
            td.textContent = cell;
            tr.appendChild(td);
        });
        tbody.appendChild(tr);
    });
});
