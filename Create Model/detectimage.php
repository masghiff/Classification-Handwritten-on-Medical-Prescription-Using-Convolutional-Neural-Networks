<?php
     include "admin/database.php";
     $filename="";

     $result=["gambar"=>"","class"=>""];
     if (isset($_FILES["imageFile"]))
     {
         $ft2="temp.jpg";
         $ftarget='gambar/'. $ft2;
         if (move_uploaded_file($_FILES["imageFile"]['tmp_name'],$ftarget)) {
             //echo "Uploaded";
             $filename=$ft2;
             $command="python predict.py";
             $output = explode("######",shell_exec($command));
             $class=$output[count($output)-1];
             $result["class"]=str_replace("\n","",trim($class));
             $gambar="";
             //if ($class!="Tidak terdeteksi")
             //{
             $q="SELECT * FROM obat WHERE Nama='".strtolower($result["class"])."'";
             $res=mysqli_query($conn,$q);
             if ($row=mysqli_fetch_assoc($res))
             {
                $result["gambar"]=$row["Gambar"];
             }

             //}
             //echo $output[count($output)-1];
         } else {
         }
     }

     echo json_encode($result);
?>