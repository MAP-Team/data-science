using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class calc_real_dist : MonoBehaviour
{
    public GameObject cube;
    public GameObject cameraRight;
    public GameObject cameraLeft;

    // Start is called before the first frame update
    void Start()
    {
    }

    // Update is called once per frame
    void Update()
    {
        Vector3 midpoint = (cameraLeft.transform.position + cameraRight.transform.position)/2;
        float dist = Vector3.Distance(cube.transform.position, midpoint);
        Debug.Log(dist);
    }
}
