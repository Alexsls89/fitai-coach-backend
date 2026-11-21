import 'dart:async';
import 'dart:convert';
import 'package:flutter/foundation.dart';
import 'package:http/http.dart' as http;
import 'package:http_parser/http_parser.dart';
import 'package:image_picker/image_picker.dart';
import '../core/api_config.dart';

class BodyVisionService {
  static bool _isTransientStatus(int code) {
    return code == 408 || code == 429 || code == 502 || code == 503 || code == 504;
  }

  static Future> analyzeBody360({
    required XFile frontImage,
    XFile? backImage,
    XFile? leftImage,
    XFile? rightImage,
    String userEmail = 'aluno_padrao',
    String userName = 'Aluno FitAI',
    double? weightKg,
    double? heightCm,
    String goal = 'Hipertrofia & Definição',
    String level = 'Intermediário',
    int daysPerWeek = 4,
    String restrictions = 'Nenhuma restrição informada',
    bool usesMounjaro = false,
    String? mounjaroDose,
    String? previousEvaluations,
  }) async {
    final frontBytes = await frontImage.readAsBytes();
    final backBytes = backImage != null ? await backImage.readAsBytes() : null;
    final leftBytes = leftImage != null ? await leftImage.readAsBytes() : null;
    final rightBytes = rightImage != null ? await rightImage.readAsBytes() : null;

    Future<http.Response> sendAttempt() async {
      final uri = Uri.parse('${ApiConfig.baseUrl}/analyze-body-360');
      final request = http.MultipartRequest('POST', uri);

      void attachBytes(Uint8List? bytes, XFile? file, String fieldName) {
        if (bytes == null || bytes.isEmpty) return;
        request.files.add(
          http.MultipartFile.fromBytes(
            fieldName,
            bytes,
            filename: (file != null && file.name.isNotEmpty)
                ? file.name
                : '$fieldName.jpg',
            contentType: MediaType('image', 'jpeg'),
          ),
        );
      }

      attachBytes(frontBytes, frontImage, 'front_image');
      attachBytes(backBytes, backImage, 'back_image');
      attachBytes(leftBytes, leftImage, 'left_image');
      attachBytes(rightBytes, rightImage, 'right_image');

      request.fields['user_email'] = userEmail.trim().isEmpty
          ? 'aluno_padrao'
          : userEmail.trim().toLowerCase();
      request.fields['user_name'] =
          userName.trim().isEmpty ? 'Aluno FitAI' : userName.trim();

      if (weightKg != null) request.fields['weight_kg'] = weightKg.toString();
      if (heightCm != null) request.fields['height_cm'] = heightCm.toString();
      request.fields['goal'] = goal;
      request.fields['level'] = level;
      request.fields['days_per_week'] = daysPerWeek.toString();
      request.fields['restrictions'] = restrictions;
      request.fields['uses_mounjaro'] = usesMounjaro.toString();
      if (mounjaroDose != null && mounjaroDose.isNotEmpty) {
        request.fields['mounjaro_dose'] = mounjaroDose;
      }
      if (previousEvaluations != null && previousEvaluations.isNotEmpty) {
        request.fields['previous_evaluations'] = previousEvaluations;
      }

      final streamed = await request.send().timeout(const Duration(seconds: 90));
      return http.Response.fromStream(streamed);
    }

    http.Response? lastResponse;
    Object? lastError;

    for (int attempt = 1; attempt <= 2; attempt++) {
      try {
        final resp = await sendAttempt();
        if (resp.statusCode >= 200 && resp.statusCode < 300) {
          return Map<String, dynamic>.from(
            jsonDecode(utf8.decode(resp.bodyBytes)) as Map,
          );
        }
        lastResponse = resp;
        if (!_isTransientStatus(resp.statusCode) || attempt == 2) {
          break;
        }
        await Future.delayed(const Duration(seconds: 2));
      } catch (e) {
        lastError = e;
        if (attempt == 2) break;
        await Future.delayed(const Duration(seconds: 2));
      }
    }

    if (lastResponse != null) {
      throw Exception(
        'Erro servidor (${lastResponse.statusCode}): ${utf8.decode(lastResponse.bodyBytes)}',
      );
    }
    throw Exception('Falha de conexão com o servidor: $lastError');
  }

  static Future> fetchBodyHistory({
    required String userEmail,
  }) async {
    final cleanEmail = userEmail.trim().isEmpty
        ? 'aluno_padrao'
        : userEmail.trim().toLowerCase();
    final uri = Uri.parse('${ApiConfig.baseUrl}/body-vision/history').replace(
      queryParameters: {'user_email': cleanEmail},
    );

    final response = await http.get(uri).timeout(const Duration(seconds: 25));
    if (response.statusCode >= 200 && response.statusCode < 300) {
      return Map<String, dynamic>.from(
        jsonDecode(utf8.decode(response.bodyBytes)) as Map,
      );
    }
    throw Exception('Erro ao carregar histórico (${response.statusCode})');
  }

  static Future<List>> fetchCoachStudents() async {
    final uri = Uri.parse('${ApiConfig.baseUrl}/coach/students');
    final response = await http.get(uri).timeout(const Duration(seconds: 25));
    if (response.statusCode >= 200 && response.statusCode < 300) {
      final decoded = Map<String, dynamic>.from(
        jsonDecode(utf8.decode(response.bodyBytes)) as Map,
      );
      final list = decoded['students'];
      if (list is List) {
        return list
            .whereType()
            .map((e) => Map<String, dynamic>.from(e))
            .toList();
      }
      return [];
    }
    throw Exception('Erro ao listar alunos (${response.statusCode})');
  }

  static Future> generateCustomWorkout({
    required String goal,
    required String level,
    required int daysPerWeek,
    required String restrictions,
  }) async {
    final uri = Uri.parse('${ApiConfig.baseUrl}/generate-workout');

    for (int attempt = 1; attempt <= 2; attempt++) {
      try {
        final response = await http
            .post(
              uri,
              body: {
                'goal': goal,
                'level': level,
                'days_per_week': daysPerWeek.toString(),
                'restrictions': restrictions.trim().isEmpty
                    ? 'Nenhuma restrição'
                    : restrictions.trim(),
              },
            )
            .timeout(const Duration(seconds: 65));

        if (response.statusCode >= 200 && response.statusCode < 300) {
          return Map<String, dynamic>.from(
            jsonDecode(utf8.decode(response.bodyBytes)) as Map,
          );
        }
        if (!_isTransientStatus(response.statusCode) || attempt == 2) {
          throw Exception(
            'Erro ao gerar planilha (${response.statusCode}): ${utf8.decode(response.bodyBytes)}',
          );
        }
        await Future.delayed(const Duration(seconds: 2));
      } catch (e) {
        if (attempt == 2) rethrow;
        await Future.delayed(const Duration(seconds: 2));
      }
    }
    throw Exception('Não foi possível gerar a planilha agora.');
  }

  static Future> analyzeMealPhoto({
    required XFile mealImage,
    String goal = 'Hipertrofia & Definição',
  }) async {
    final bytes = await mealImage.readAsBytes();

    for (int attempt = 1; attempt <= 2; attempt++) {
      try {
        final uri = Uri.parse('${ApiConfig.baseUrl}/analyze-meal');
        final request = http.MultipartRequest('POST', uri);

        request.files.add(
          http.MultipartFile.fromBytes(
            'meal_image',
            bytes,
            filename: mealImage.name.isNotEmpty ? mealImage.name : 'meal.jpg',
            contentType: MediaType('image', 'jpeg'),
          ),
        );
        request.fields['goal'] = goal;

        final streamed =
            await request.send().timeout(const Duration(seconds: 55));
        final response = await http.Response.fromStream(streamed);

        if (response.statusCode >= 200 && response.statusCode < 300) {
          return Map<String, dynamic>.from(
            jsonDecode(utf8.decode(response.bodyBytes)) as Map,
          );
        }
        if (!_isTransientStatus(response.statusCode) || attempt == 2) {
          if (kDebugMode) {
            debugPrint('Meal analysis error: ${response.statusCode}');
          }
          throw Exception(
              'Falha ao analisar refeição (${response.statusCode})');
        }
        await Future.delayed(const Duration(seconds: 2));
      } catch (e) {
        if (attempt == 2) rethrow;
        await Future.delayed(const Duration(seconds: 2));
      }
    }
    throw Exception('Falha ao analisar refeição.');
  }
}